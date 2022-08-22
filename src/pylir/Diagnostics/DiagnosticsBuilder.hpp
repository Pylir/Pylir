// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/Support/raw_ostream.h>

#include <variant>

#include <fmt/color.h>
#include <fmt/format.h>

#include "DiagnosticsManager.hpp"
#include "Document.hpp"
#include "LocationProvider.hpp"

namespace pylir::Diag
{

namespace flags
{
namespace detail
{

/// Payload created through assigning to a FlagParam.
template <class T, class ID>
struct FlagValue
{
    T value;

    /// Whether this FlagValue was created via assignment to 'flag'.
    template <auto& flag>
    constexpr static bool isInstanceOf()
    {
        return std::is_same_v<typename std::decay_t<decltype(flag)>::identifier, ID>;
    }
};

/// Flag parameter which allows to use a named parameter syntax for function calls.
/// Example:
/// constexpr auto flag = FlagParam(std::in_place_type<int>,[]{});
///
/// template <class... Args>
/// void foo(Args&&...args)
/// {
///     std::optional<int> flagValue = getFlag<flag>(std::forward<Args>(args)...);
///     ....
/// }
///
/// foo(flag = 3);
///
/// A flag may also not carry a value in which case it is constructed as 'FlagParam([]{})'. Whether the flag was
/// specified can then be queried via 'getFlag<Args...>()'. When a flag does carry a value it is required to be assigned
/// to, otherwise a compile time error is emitted.
///
/// The purpose of the lambda parameter is purely to create a new unique type for the sake of compile time verification.
/// It allows users of FlagParam to uniquely identify the source flag a value came from and to trigger compile time
/// errors if incompatible flags were specified.
template <class ID, class T = std::monostate>
class FlagParam
{
public:
    /// Construct a 'FlagParam' that contains the type T.
    constexpr FlagParam(std::in_place_type_t<T>, ID) {}

    /// Constructs a 'FlagParam' without a value.
    constexpr FlagParam(ID) {}

    using value_type = T;
    using identifier = ID;

    /// Assigns a value to this, creating a FlagValue in the process.
    template <class U, class V = T, std::enable_if_t<!std::is_same_v<V, std::monostate>>* = nullptr>
    constexpr FlagValue<T, ID> operator=(U&& value) const
    {
        return {std::forward<U>(value)};
    }

    /// Checks whether this 'FlagParam' is the specific 'flag'.
    template <auto& flag>
    constexpr static bool isInstanceOf()
    {
        return std::is_same_v<typename std::decay_t<decltype(flag)>::identifier, ID>;
    }
};

/// Template used to check whether 'T' is a flag. This may either then be a 'FlagParam', if it does not carry a value
/// or a 'FlagValue' if it does.
template <class T>
struct IsFlag : std::false_type
{
};

template <class ID>
struct IsFlag<flags::detail::FlagParam<ID, std::monostate>> : std::true_type
{
};

template <class ID, class T>
struct IsFlag<flags::detail::FlagParam<ID, T>> : std::true_type
{
    static_assert(sizeof(T) == 0, "Flag requires value but none was assigned via '='.");
};

template <class T, class ID>
struct IsFlag<typename flags::detail::FlagValue<T, ID>> : std::true_type
{
};

/// Function to extract the value of a given flag from a parameter pack of flags. If the flag is not contained, an empty
/// optional is returned. If the flag is specified multiple times, the last occurrences value is used.
template <auto& flag, class... Args, class ValueType = typename std::decay_t<decltype(flag)>::value_type>
constexpr std::optional<ValueType> getFlag(Args&&... args)
{
    static_assert((true && ... && IsFlag<std::decay_t<Args>>{}));
    std::optional<ValueType> result;
    (
        [&](auto&& arg)
        {
            if constexpr (std::decay_t<decltype(arg)>::template isInstanceOf<flag>())
            {
                result.emplace(std::forward<decltype(arg.value)>(arg.value));
            }
        }(args),
        ...);
    return result;
}

/// Returns true if the given flag is contained within the list of 'Args'.
template <auto& flag, class... Args>
constexpr bool hasFlag()
{
    return (std::decay_t<Args>::template isInstanceOf<flag>() || ...);
}

} // namespace detail

/// Flags used to customize the effect of 'addLabel' in DiagnosticBuilder. A flag may also carry a value of the type
/// specified in 'std::in_place_type', in which case it is required to set a value via assignment.
/// Example:
///         builder.addLabel(0, 1, flags::label = "text", flags::bold);

/// Label to be used when printing. This is plain text which appears underneath the highlight markers.
constexpr auto label = detail::FlagParam(std::in_place_type<std::string>, [] {});
/// Indicates that no colour should be used when printing the source code. By default 'primaryColour' is used to
/// highlight the source code denoted by the location given to 'addLabel'.
constexpr auto noColour = detail::FlagParam([] {});
/// Use the primary colour to highlight the source code. The precise colour is determined by the kind of message the
/// 'addLabel' call is being attached to.
/// This is the default.
constexpr auto primaryColour = detail::FlagParam([] {});
/// Use the secondary colour to highlight the source code. This colour is meant to complement the primary colour and
/// used to highlight things that aren't off as high of an importance as the primary colour.
constexpr auto secondaryColour = detail::FlagParam([] {});
/// Colour used to indicate to the user that the given label is meant as an suggestion how to fix their code.
constexpr auto insertColour = detail::FlagParam([] {});
/// Modifies the text the label is highlighting to be struck through. Used to indicate parts of the source code that
/// should be removed.
constexpr auto strikethrough = detail::FlagParam([] {});
/// Modifies the text the label is highlighting to be displayed in a bolder font.
constexpr auto bold = detail::FlagParam([] {});
/// Modifies the text the label is highlighting to be displayed as italic.
constexpr auto italic = detail::FlagParam([] {});

} // namespace flags

/// Severity of the message. Printed as part of a message and affects the primary and secondary colours used for
/// highlighting.
enum class Severity
{
    Warning,
    Error,
    Note
};

/// Base class for 'DiagnosticsBuilder' containing data and methods that are independent of the diagnostics manager
/// used.
class DiagnosticsBuilderBase
{
    llvm::raw_ostream& (*m_emitFn)(llvm::raw_ostream&, const DiagnosticsBuilderBase&);

public:
    constexpr static auto ERROR_COLOUR = fmt::color::red;
    constexpr static auto WARNING_COLOUR = fmt::color::magenta;
    constexpr static auto NOTE_COLOUR = fmt::color::cyan;

protected:
    struct Label
    {
        std::size_t start;
        std::size_t end;
        std::optional<std::string> labelText;
        std::optional<fmt::color> optionalColour;
        std::optional<fmt::emphasis> optionalEmphasis;
    };

    struct Message
    {
        Severity severity;
        std::size_t location;
        std::string message;
        std::vector<Label> labels;
    };
    std::vector<Message> m_messages;

    static void printLine(llvm::raw_ostream& os, std::size_t width, std::size_t lineNumber,
                          const pylir::Diag::Document& document, std::vector<Label> labels);

    void emitMessage(llvm::raw_ostream& os, const Message& message,
                     Diag::DiagnosticsDocManager* diagnosticDocManager) const;

    template <class T>
    explicit DiagnosticsBuilderBase(Message&& message, std::in_place_type_t<T>)
        : m_emitFn(+[](llvm::raw_ostream& os, const DiagnosticsBuilderBase& ptr) -> llvm::raw_ostream&
                   { return os << static_cast<const T&>(ptr); })
    {
        m_messages.push_back(std::move(message));
    }

public:
    /// Returns the main message of the diagnostic. This is the message that is initialized in the constructor of
    /// 'DiagnosticsBuilder' and is the only one that may be a warning or error.
    [[nodiscard]] const Message& getMainDiagnostic() const
    {
        return m_messages.front();
    }

    /// Outputs this diagnostic to an output stream. Uses ASCII escape for printing colour and emphasis if the output
    /// stream supports it.
    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const DiagnosticsBuilderBase& base)
    {
        return base.m_emitFn(os, base);
    }
};

/// Class used to build a compiler diagnostic. It makes use of the builder pattern to be able to conveniently chain
/// a series of method calls, modifying the result. The final diagnostic is the emitted upon destruction.
///
/// A diagnostic builder may either be instantiated with a 'DiagnosticsDocManager' in which case it requires the use of
/// locations into the document for source file location, or with a 'DiagnosticsNoDocManager', in which case it only
/// supports the output of the messages and severity set in the the constructor and 'addNote'.
///
/// While building a diagnostic in a document it consists of the initial header message set in the constructor, its
/// severity and is then followed by output of source code. The printing of the source code can further be customized
/// via calls to 'addLabel' which allow to underline, colour, label as well as modify the character style used to print
/// portions of the source code. The source code to be printed is determined via the locations passed to the constructor
/// as well as any locations passed to 'addLabel'.
///
/// Following the initial diagnostic, one can also attach additional notes via 'addNote'. These follow the same
/// formatting as errors and warnings described above. 'addLabel' calls always affect the last diagnostic added,
/// including notes.
template <class Manager>
class DiagnosticsBuilder : public DiagnosticsBuilderBase
{
    Manager* m_diagnosticManagerBase = nullptr;
    template <class... Args>
    static void verify()
    {
        constexpr bool hasPrimary = flags::detail::hasFlag<flags::primaryColour, Args...>();
        constexpr bool hasSecondary = flags::detail::hasFlag<flags::secondaryColour, Args...>();
        constexpr bool hasInsert = flags::detail::hasFlag<flags::insertColour, Args...>();
        constexpr bool hasNoColour = flags::detail::hasFlag<flags::noColour, Args...>();
        static_assert(hasPrimary + hasSecondary + hasInsert + hasNoColour <= 1,
                      "Only one of primary, secondary, no colour or insert colour may be specified.");

        static_assert(flags::detail::hasFlag<flags::strikethrough, Args...>()
                              + flags::detail::hasFlag<flags::bold, Args...>()
                              + flags::detail::hasFlag<flags::italic, Args...>()
                          <= 1,
                      "Only one of strikethrough, bold or italic may be specified.");
    }

    template <class... Args>
    std::optional<fmt::color> getColour()
    {
        if constexpr (flags::detail::hasFlag<flags::secondaryColour, Args...>())
        {
            switch (m_messages.back().severity)
            {
                case Severity::Error: return fmt::color::orange;
                case Severity::Warning: return fmt::color::plum;
                case Severity::Note: return fmt::color::spring_green;
            }
        }
        else if constexpr (flags::detail::hasFlag<flags::insertColour, Args...>())
        {
            return fmt::color::lime_green;
        }
        else if constexpr (flags::detail::hasFlag<flags::noColour, Args...>())
        {
            return std::nullopt;
        }
        switch (m_messages.back().severity)
        {
            case Severity::Error: return ERROR_COLOUR;
            case Severity::Warning: return WARNING_COLOUR;
            case Severity::Note: return NOTE_COLOUR;
        }
        PYLIR_UNREACHABLE;
    }

    template <class... Args>
    std::optional<fmt::emphasis> getEmphasis(Args&... args)
    {
        std::optional<fmt::emphasis> emphasis;
        (
            [&](auto& arg)
            {
                using Arg = std::decay_t<decltype(arg)>;
                if constexpr (Arg::template isInstanceOf<flags::strikethrough>())
                {
                    emphasis = fmt::emphasis::strikethrough;
                }
                if constexpr (Arg::template isInstanceOf<flags::bold>())
                {
                    emphasis = fmt::emphasis::bold;
                }
                if constexpr (Arg::template isInstanceOf<flags::italic>())
                {
                    emphasis = fmt::emphasis::italic;
                }
            }(args),
            ...);
        return emphasis;
    }

public:
    /// Creates a new DiagnosticBuilder for a message with the given 'severity' with a diagnostic at 'location' within
    /// the document of 'subDiagnosticManager'. 'location' may be any type that can be used as a location via
    /// 'rangeLoc' with the context in 'subDiagnosticManager'. See its documentation for details.
    ///
    /// The severity specifies the kind of message this diagnostic is producing. This is used as prefix when printing
    /// and for the colours to use as primary and secondary colours.
    /// The actual text message printed is produced via 'message', which may use 'fmt::format' style formatting options
    /// with 'args' as extra arguments.
    template <class T, class S, class... Args>
    DiagnosticsBuilder(DiagnosticsDocManager& subDiagnosticManager, Severity severity, const T& location,
                       const S& message, Args&&... args)
        : DiagnosticsBuilderBase({severity,
                                  rangeLoc(location, subDiagnosticManager.getContext()).first,
                                  fmt::format(message, std::forward<Args>(args)...),
                                  {}},
                                 std::in_place_type<std::decay_t<decltype(*this)>>),
          m_diagnosticManagerBase(&subDiagnosticManager)
    {
    }

    /// Creates a new DiagnosticBuilder for a message with the given 'severity'.
    /// The severity specifies the kind of message this diagnostic is producing. This is used as prefix when printing
    /// and for the colours to use. The actual text message printed is produced via 'message', which may use
    /// 'fmt::format' style formatting options with 'args' as extra arguments.
    template <class S, class... Args>
    DiagnosticsBuilder(DiagnosticsNoDocManager& subDiagnosticManager, Severity severity, const S& message,
                       Args&&... args)
        : DiagnosticsBuilderBase({severity, 0, fmt::format(message, std::forward<Args>(args)...), {}},
                                 std::in_place_type<std::decay_t<decltype(*this)>>),
          m_diagnosticManagerBase(&subDiagnosticManager)
    {
    }

    /// Emits the final diagnostic by reporting it to the 'subDiagnosticManager' it was constructed with.
    ~DiagnosticsBuilder()
    {
        if (!m_diagnosticManagerBase)
        {
            return;
        }
        m_diagnosticManagerBase->report(std::move(*this));
    }

    DiagnosticsBuilder(const DiagnosticsBuilder&) = delete;
    DiagnosticsBuilder& operator=(const DiagnosticsBuilder&) = delete;

    DiagnosticsBuilder(DiagnosticsBuilder&& rhs) noexcept
        : DiagnosticsBuilderBase(std::move(rhs)),
          m_diagnosticManagerBase(std::exchange(rhs.m_diagnosticManagerBase, nullptr))
    {
    }

    DiagnosticsBuilder& operator=(DiagnosticsBuilder&& rhs) noexcept
    {
        *this = static_cast<DiagnosticsBuilderBase&&>(rhs);
        m_diagnosticManagerBase = std::exchange(rhs.m_diagnosticManagerBase, nullptr);
        return *this;
    }

    /// Adds the effects denoted by the flags to the given source code ranging from 'start' until the very end of 'end'.
    template <class T, class U, class... Flags, class M = Manager>
    auto addLabel(const T& start, const U& end, Flags&&... flags)
        -> std::enable_if_t<hasLocationProvider_v<T> && hasLocationProvider_v<U>
                                && std::conjunction_v<flags::detail::IsFlag<std::decay_t<Flags>>...>
                                && std::is_same_v<M, DiagnosticsDocManager>,
                            DiagnosticsBuilder&&>
    {
        verify<Flags...>();
        m_messages.back().labels.push_back({rangeLoc(start, m_diagnosticManagerBase->getContext()).first,
                                            rangeLoc(end, m_diagnosticManagerBase->getContext()).second,
                                            flags::detail::getFlag<flags::label>(std::forward<Flags>(flags)...),
                                            getColour<Flags...>(), getEmphasis(flags...)});
        return std::move(*this);
    }

    /// Adds the effects denoted by the flags to the given source code at 'pos'.
    template <class T, class... Flags, class M = Manager>
    auto addLabel(const T& pos, Flags&&... flags)
        -> std::enable_if_t<hasLocationProvider_v<T>
                                && std::conjunction_v<flags::detail::IsFlag<std::decay_t<Flags>>...>
                                && std::is_same_v<M, DiagnosticsDocManager>,
                            DiagnosticsBuilder&&>
    {
        verify<Flags...>();
        auto [start, end] = rangeLoc(pos, m_diagnosticManagerBase->getContext());
        m_messages.back().labels.push_back({start, end,
                                            flags::detail::getFlag<flags::label>(std::forward<Flags>(flags)...),
                                            getColour<Flags...>(), getEmphasis(flags...)});
        return std::move(*this);
    }

    /// Adds the effects denoted by the flags to the given source code ranging from 'start' until the very end of 'end'.
    /// This overload has an explicit parm for the label as it is a commonly used option.
    template <class T, class U, class... Flags, class M = Manager>
    auto addLabel(const T& start, const U& end, std::string label, Flags&&... flags)
        -> std::enable_if_t<hasLocationProvider_v<T> && hasLocationProvider_v<U>
                                && std::conjunction_v<flags::detail::IsFlag<std::decay_t<Flags>>...>
                                && std::is_same_v<M, DiagnosticsDocManager>,
                            DiagnosticsBuilder&&>
    {
        return addLabel(start, end, flags::label = std::move(label), std::forward<Flags>(flags)...);
    }

    /// Adds the effects denoted by the flags to the given source code at 'pos'.
    /// This overload has an explicit parm for the label as it is a commonly used option.
    template <class T, class... Flags, class M = Manager>
    auto addLabel(const T& pos, std::string label, Flags&&... flags)
        -> std::enable_if_t<hasLocationProvider_v<T>
                                && std::conjunction_v<flags::detail::IsFlag<std::decay_t<Flags>>...>
                                && std::is_same_v<M, DiagnosticsDocManager>,
                            DiagnosticsBuilder&&>
    {
        return addLabel(pos, flags::label = std::move(label), std::forward<Flags>(flags)...);
    }

    /// Adds a new note to the diagnostic. 'location' as well as 'message' and its 'args' serve the same purpose as in
    /// the constructor. The document and context are taken from the 'DiagnosticsDocManager' it was initially
    /// constructed with.
    template <class T, class S, class... Args, class M = Manager>
    auto addNote(const T& location, const S& message, Args&&... args)
        -> std::enable_if_t<hasLocationProvider_v<T> && std::is_same_v<M, DiagnosticsDocManager>, DiagnosticsBuilder&&>
    {
        m_messages.push_back({Severity::Note,
                              rangeLoc(location, m_diagnosticManagerBase->getContext()).first,
                              fmt::format(message, std::forward<Args>(args)...),
                              {}});
        return std::move(*this);
    }

    /// Adds a new note to the diagnostic. 'message' and its 'args' serve the same purpose as in
    /// the constructor.
    template <class S, class... Args, class M = Manager>
    auto addNote(const S& message, Args&&... args)
        -> std::enable_if_t<std::is_same_v<M, DiagnosticsNoDocManager>, DiagnosticsBuilder&&>
    {
        m_messages.push_back({Severity::Note, 0, fmt::format(message, std::forward<Args>(args)...), {}});
        return std::move(*this);
    }

    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const DiagnosticsBuilder& rhs)
    {
        DiagnosticsDocManager* docManager = nullptr;
        if constexpr (std::is_same_v<Manager, DiagnosticsDocManager>)
        {
            docManager = rhs.m_diagnosticManagerBase;
        }
        for (auto& iter : rhs.m_messages)
        {
            rhs.emitMessage(os, iter, docManager);
        }
        return os;
    }
};

template <class T, class S, class... Args>
DiagnosticsBuilder(DiagnosticsDocManager&, Severity, const T&, const S&, Args&&...)
    -> DiagnosticsBuilder<DiagnosticsDocManager>;

template <class S, class... Args>
DiagnosticsBuilder(DiagnosticsNoDocManager&, Severity, const S&, Args&&...)
    -> DiagnosticsBuilder<DiagnosticsNoDocManager>;

} // namespace pylir::Diag
