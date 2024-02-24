//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>

#include <pylir/Support/AbstractIntrusiveVariant.hpp>

#include <variant>

#include <fmt/color.h>
#include <fmt/format.h>

#include "DiagnosticsManager.hpp"
#include "Document.hpp"
#include "LocationProvider.hpp"

namespace pylir::Diag {

namespace flags {
namespace detail {

/// Payload created through assigning to a FlagParam.
template <class T, class ID>
struct FlagValue {
  T value;

  /// Whether this FlagValue was created via assignment to 'flag'.
  template <auto& flag>
  constexpr static bool isInstanceOf() {
    return std::is_same_v<typename std::decay_t<decltype(flag)>::identifier,
                          ID>;
  }
};

/// Flag parameter which allows to use a named parameter syntax for function
/// calls. Example:
/// constexpr auto flag = FlagParam(std::in_place_type<int>,[]{});
///
/// template <class... Args>
/// void foo(Args&&...args) {
///     std::optional<int> flagValue =
///     getFlag<flag>(std::forward<Args>(args)...);
///     ....
/// }
///
/// foo(flag = 3);
///
/// A flag may also not carry a value in which case it is constructed as
/// 'FlagParam([]{})'. Whether the flag was specified can then be queried via
/// 'getFlag<Args...>()'. When a flag does carry a value it is required to be
/// assigned to, otherwise a compile time error is emitted.
///
/// The purpose of the lambda parameter is purely to create a new unique type
/// for the sake of compile time verification. It allows users of FlagParam to
/// uniquely identify the source flag a value came from and to trigger compile
/// time errors if incompatible flags were specified.
template <class ID, class T = std::monostate>
class FlagParam {
public:
  /// Construct a 'FlagParam' that contains the type T.
  constexpr FlagParam(std::in_place_type_t<T>, ID) {}

  /// Constructs a 'FlagParam' without a value.
  constexpr explicit FlagParam(ID) {}

  using value_type = T;
  using identifier = ID;

  /// Assigns a value to this, creating a FlagValue in the process.
  template <class U, class V = T,
            std::enable_if_t<!std::is_same_v<V, std::monostate>>* = nullptr>
  constexpr FlagValue<T, ID> operator=(U&& value) const {
    return {std::forward<U>(value)};
  }

  /// Checks whether this 'FlagParam' is the specific 'flag'.
  template <auto& flag>
  constexpr static bool isInstanceOf() {
    return std::is_same_v<typename std::decay_t<decltype(flag)>::identifier,
                          ID>;
  }
};

/// Template used to check whether 'T' is a flag. This may either then be a
/// 'FlagParam', if it does not carry a value or a 'FlagValue' if it does.
template <class T>
struct IsFlag : std::false_type {};

template <class ID>
struct IsFlag<flags::detail::FlagParam<ID, std::monostate>> : std::true_type {};

template <class ID, class T>
struct IsFlag<flags::detail::FlagParam<ID, T>> : std::true_type {
  static_assert(sizeof(T) == 0,
                "Flag requires value but none was assigned via '='.");
};

template <class T, class ID>
struct IsFlag<typename flags::detail::FlagValue<T, ID>> : std::true_type {};

/// Function to extract the value of a given flag from a parameter pack of
/// flags. If the flag is not contained, an empty optional is returned. If the
/// flag is specified multiple times, the last occurrences value is used.
template <auto& flag, class... Args,
          class ValueType = typename std::decay_t<decltype(flag)>::value_type>
constexpr std::optional<ValueType> getFlag(Args&&... args) {
  static_assert((true && ... && IsFlag<std::decay_t<Args>>{}));
  std::optional<ValueType> result;
  (
      [&](auto&& arg) {
        if constexpr (std::decay_t<decltype(arg)>::template isInstanceOf<
                          flag>())
          result.emplace(std::forward<decltype(arg.value)>(arg.value));
      }(args),
      ...);
  return result;
}

/// Returns true if the given flag is contained within the list of 'Args'.
template <auto& flag, class... Args>
constexpr bool hasFlag() {
  return (std::decay_t<Args>::template isInstanceOf<flag>() || ...);
}

} // namespace detail

/// Flags used to customize the effect of 'addHighlight' in DiagnosticBuilder. A
/// flag may also carry a value of the type specified in 'std::in_place_type',
/// in which case it is required to set a value via assignment.
/// Example: builder.addHighlight(0, 1, flags::label = "text", flags::bold);

/// Label to be used when printing. This is plain text which appears underneath
/// the highlight markers.
constexpr auto label =
    detail::FlagParam(std::in_place_type<std::string>, [] {});
/// Indicates that no colour should be used when printing the source code. By
/// default 'primaryColour' is used to highlight the source code denoted by the
/// location given to 'addHighlight'.
constexpr auto noColour = detail::FlagParam([] {});
/// Use the primary colour to highlight the source code. The precise colour is
/// determined by the kind of message the 'addHighlight' call is being attached
/// to. This is the default.
constexpr auto primaryColour = detail::FlagParam([] {});
/// Use the secondary colour to highlight the source code. This colour is meant
/// to complement the primary colour and used to highlight things that aren't
/// off as high of an importance as the primary colour.
constexpr auto secondaryColour = detail::FlagParam([] {});
/// Colour used to indicate to the user that the given label is meant as an
/// suggestion how to fix their code.
constexpr auto insertColour = detail::FlagParam([] {});
/// Modifies the text the label is highlighting to be struck through. Used to
/// indicate parts of the source code that should be removed.
constexpr auto strikethrough = detail::FlagParam([] {});
/// Modifies the text the label is highlighting to be displayed in a bolder
/// font.
constexpr auto bold = detail::FlagParam([] {});
/// Modifies the text the label is highlighting to be displayed as italic.
constexpr auto italic = detail::FlagParam([] {});

} // namespace flags

/// Severity of the message. Printed as part of a message and affects the
/// primary and secondary colours used for highlighting.
enum class Severity { Warning, Error, Note };

/// Container struct containing all info related to a single diagnostic.
/// Consists of multiple messages, which usually is one warning or error,
/// possibly followed by more notes.
struct Diagnostic {
  constexpr static auto ERROR_COLOUR = fmt::color::red;
  constexpr static auto WARNING_COLOUR = fmt::color::magenta;
  constexpr static auto NOTE_COLOUR = fmt::color::cyan;

  struct Highlight {
    std::size_t start{};
    std::size_t end{};
    std::optional<std::string> highlightText;
    std::optional<fmt::color> optionalColour;
    std::optional<fmt::emphasis> optionalEmphasis;
  };

  struct Message {
    Severity severity;
    const Document* document; // may be null if there is no proper location
    std::size_t location;     // meaningless if document is null
    std::string message;
    std::vector<Highlight> highlights;
  };

  std::vector<Message> messages;

private:
  static void printLine(llvm::raw_ostream& os, std::size_t width,
                        std::size_t lineNumber,
                        const pylir::Diag::Document& document,
                        std::vector<Highlight> highlights);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const Message& message);

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const Diagnostic& rhs) {
    for (const auto& iter : rhs.messages)
      os << iter;

    return os;
  }
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const Diagnostic::Message& message);

/// Class used to build a compiler diagnostic. It makes use of the builder
/// pattern to be able to conveniently chain a series of method calls, modifying
/// the result. The final diagnostic may then be emitted upon destruction.
///
/// A diagnostic builder may either be instantiated with a
/// 'DiagnosticsDocManager' in which case it requires the use of locations into
/// the document for source file location, or with a 'DiagnosticsNoDocManager',
/// in which case it only supports the output of the messages and severity set
/// in the the constructor and 'addNote'.
///
/// While building a diagnostic in a document it consists of the initial header
/// message set in the constructor, its severity and is then followed by output
/// of source code. The printing of the source code can further be customized
/// via calls to 'addHighlight' which allow to underline, colour, label as well
/// as modify the character style used to print portions of the source code. The
/// source code to be printed is determined via the locations passed to the
/// constructor as well as any locations passed to 'addHighlight'.
///
/// Following the initial diagnostic, one can also attach additional notes via
/// 'addNote'. These follow the same formatting as errors and warnings described
/// above. 'addHighlight' calls always affect the last diagnostic added,
/// including notes.
template <class Manager>
class DiagnosticsBuilder {
  Diagnostic m_diagnostic;
  Manager* m_diagnosticManagerBase = nullptr;

  template <class... Args>
  static void verify() {
    constexpr bool hasPrimary =
        flags::detail::hasFlag<flags::primaryColour, Args...>();
    constexpr bool hasSecondary =
        flags::detail::hasFlag<flags::secondaryColour, Args...>();
    constexpr bool hasInsert =
        flags::detail::hasFlag<flags::insertColour, Args...>();
    constexpr bool hasNoColour =
        flags::detail::hasFlag<flags::noColour, Args...>();
    static_assert(hasPrimary + hasSecondary + hasInsert + hasNoColour <= 1,
                  "Only one of primary, secondary, no colour or insert colour "
                  "may be specified.");

    static_assert(
        flags::detail::hasFlag<flags::strikethrough, Args...>() +
                flags::detail::hasFlag<flags::bold, Args...>() +
                flags::detail::hasFlag<flags::italic, Args...>() <=
            1,
        "Only one of strikethrough, bold or italic may be specified.");
  }

  template <class... Args>
  std::optional<fmt::color> getColour() {
    if constexpr (flags::detail::hasFlag<flags::secondaryColour, Args...>()) {
      switch (m_diagnostic.messages.back().severity) {
      case Severity::Error: return fmt::color::orange;
      case Severity::Warning: return fmt::color::plum;
      case Severity::Note: return fmt::color::spring_green;
      }
    } else if constexpr (flags::detail::hasFlag<flags::insertColour,
                                                Args...>()) {
      return fmt::color::lime_green;
    } else if constexpr (flags::detail::hasFlag<flags::noColour, Args...>())
      return std::nullopt;
    switch (m_diagnostic.messages.back().severity) {
    case Severity::Error: return Diagnostic::ERROR_COLOUR;
    case Severity::Warning: return Diagnostic::WARNING_COLOUR;
    case Severity::Note: return Diagnostic::NOTE_COLOUR;
    }
    PYLIR_UNREACHABLE;
  }

  template <class... Args>
  std::optional<fmt::emphasis> getEmphasis(Args&... args) {
    std::optional<fmt::emphasis> emphasis;
    (
        [&](auto& arg) {
          using Arg = std::decay_t<decltype(arg)>;
          if constexpr (Arg::template isInstanceOf<flags::strikethrough>())
            emphasis = fmt::emphasis::strikethrough;
          if constexpr (Arg::template isInstanceOf<flags::bold>())
            emphasis = fmt::emphasis::bold;
          if constexpr (Arg::template isInstanceOf<flags::italic>())
            emphasis = fmt::emphasis::italic;
        }(args),
        ...);
    return emphasis;
  }

  struct IsDiagnosticsDocManagerBuilder {
    template <class T>
    std::true_type operator()(const DiagnosticsDocManager<T>&) {
      return {};
    }

    std::false_type operator()(const DiagnosticsNoDocManager&) {
      return {};
    }
  };

  constexpr static bool isDiagnosticsDocManagerBuilder() {
    return decltype(IsDiagnosticsDocManagerBuilder{}(
        std::declval<Manager>())){};
  }

  constexpr static bool hasContext() {
    return isDiagnosticsDocManagerBuilder() &&
           !std::is_same_v<Manager, DiagnosticsDocManager<void>>;
  }

  template <class T>
  static T getContextType(const DiagnosticsDocManager<T>&);

  static void getContextType(const DiagnosticsNoDocManager&);

  using Context = decltype(getContextType(std::declval<const Manager&>()));

  template <class T>
  auto rangeLoc(const T& value) const {
    if constexpr (hasContext())
      return Diag::rangeLoc(value, m_diagnosticManagerBase->getContext());
    else
      return Diag::rangeLoc(value);
  }

  template <class... Args>
  constexpr static bool areLocationProviders =
      (true && ... && hasLocationProvider_v<Args, Context>);

public:
  /// Creates a new DiagnosticBuilder for a message with the given 'severity'
  /// with a diagnostic at 'location' within the document of
  /// 'subDiagnosticManager'. 'location' may be any type that can be used as a
  /// location via 'rangeLoc' with the context in 'subDiagnosticManager'. See
  /// 'rangeLoc's documentation for details.
  ///
  /// The severity specifies the kind of message this diagnostic is producing.
  /// This is used as prefix when printing and for the colours to use as primary
  /// and secondary colours. The actual text message printed is produced via
  /// 'message', which may use 'fmt::format' style formatting options with
  /// 'args' as extra arguments.
  ///
  /// Upon destruction, the final 'Diagnostic' will be reported to
  /// 'subDiagnosticManager'.
  template <class Context, class T, class S, class... Args>
  DiagnosticsBuilder(DiagnosticsDocManager<Context>& subDiagnosticManager,
                     Severity severity, const T& location, const S& message,
                     Args&&... args)
      : m_diagnosticManagerBase(&subDiagnosticManager) {
    Location loc = rangeLoc(location);
    m_diagnostic = {{Diagnostic::Message{
        severity,
        loc ? &subDiagnosticManager.getDocument() : nullptr,
        loc ? loc->first : 0,
        fmt::format(message, std::forward<Args>(args)...),
        {}}}};
  }

  template <class T, class S, class... Args>
  DiagnosticsBuilder(DiagnosticsDocManager<void>& subDiagnosticManager,
                     Severity severity, const T& location, const S& message,
                     Args&&... args)
      : DiagnosticsBuilder(subDiagnosticManager.getDocument(), severity,
                           location, message, std::forward<Args>(args)...) {
    m_diagnosticManagerBase = &subDiagnosticManager;
  }

  /// Like the constructor above, but does not report any diagnostic upon
  /// destruction.
  template <class T, class S, class... Args>
  DiagnosticsBuilder(const Document& doc, Severity severity, const T& location,
                     const S& message, Args&&... args)
      : m_diagnosticManagerBase(nullptr) {
    Location loc = rangeLoc(location);
    m_diagnostic = {
        {Diagnostic::Message{severity,
                             loc ? &doc : nullptr,
                             loc ? loc->first : 0,
                             fmt::format(message, std::forward<Args>(args)...),
                             {}}}};
  }

  /// Creates a new DiagnosticBuilder for a message with the given 'severity'.
  /// The severity specifies the kind of message this diagnostic is producing.
  /// This is used as prefix when printing and for the colours to use. The
  /// actual text message printed is produced via 'message', which may use
  /// 'fmt::format' style formatting options with 'args' as extra arguments.
  ///
  /// Upon destruction, the final 'Diagnostic' will be reported to
  /// 'subDiagnosticManager'.
  template <class S, class... Args>
  DiagnosticsBuilder(DiagnosticsNoDocManager& subDiagnosticManager,
                     Severity severity, const S& message, Args&&... args)
      : m_diagnostic{{Diagnostic::Message{
            severity,
            nullptr,
            0,
            fmt::format(message, std::forward<Args>(args)...),
            {}}}},
        m_diagnosticManagerBase(&subDiagnosticManager) {}

  /// Emits the final diagnostic by reporting it to the 'subDiagnosticManager'
  /// it was constructed with.
  ~DiagnosticsBuilder() {
    if (!m_diagnosticManagerBase)
      return;

    m_diagnosticManagerBase->report(std::move(m_diagnostic));
  }

  /// Moves the 'Diagnostic' that was built out of the builder.
  Diagnostic&& getDiagnostic() && {
    return std::move(m_diagnostic);
  }

  DiagnosticsBuilder(const DiagnosticsBuilder&) = delete;
  DiagnosticsBuilder& operator=(const DiagnosticsBuilder&) = delete;

  DiagnosticsBuilder(DiagnosticsBuilder&& rhs) noexcept
      : m_diagnostic(std::move(rhs.m_diagnostic)),
        m_diagnosticManagerBase(
            std::exchange(rhs.m_diagnosticManagerBase, nullptr)) {}

  DiagnosticsBuilder& operator=(DiagnosticsBuilder&& rhs) noexcept {
    m_diagnostic = std::move(rhs.m_diagnostic);
    m_diagnosticManagerBase =
        std::exchange(rhs.m_diagnosticManagerBase, nullptr);
    return *this;
  }

  /// Highlights the given source code ranging from 'start' until the very end
  /// of 'end' and applies the effects denoted by the flags. Highlighting causes
  /// the given source code to be underlined with '~' if more than one character
  /// wide, or a single '^' if just a single character.
  /// Example:
  ///
  /// <stdin>:1:1: note: highlighting example:
  ///   1 | start to end dot
  ///     | ~~~~~~~~~~~~  ^
  ///
  /// where the range was 'start' token to 'end' token respectively and just the
  /// 'o' character in 'dot'.
  template <class T, class U, class... Flags>
  auto addHighlight(const T& start, const U& end, Flags&&... flags)
      -> std::enable_if_t<areLocationProviders<T, U> &&
                              std::conjunction_v<flags::detail::IsFlag<
                                  std::decay_t<Flags>>...> &&
                              isDiagnosticsDocManagerBuilder(),
                          DiagnosticsBuilder&&> {
    verify<Flags...>();
    if (Location loc = std::initializer_list<LazyLocation>{start, end})
      m_diagnostic.messages.back().highlights.push_back(
          {loc->first, loc->second,
           flags::detail::getFlag<flags::label>(std::forward<Flags>(flags)...),
           getColour<Flags...>(), getEmphasis(flags...)});
    return std::move(*this);
  }

  /// Overload of the above but with a single position instead of a range.
  template <class T, class... Flags>
  auto addHighlight(const T& pos, Flags&&... flags) -> std::enable_if_t<
      areLocationProviders<T> &&
          std::conjunction_v<flags::detail::IsFlag<std::decay_t<Flags>>...> &&
          isDiagnosticsDocManagerBuilder(),
      DiagnosticsBuilder&&> {
    verify<Flags...>();
    if (Location loc = rangeLoc(pos))
      m_diagnostic.messages.back().highlights.push_back(
          {loc->first, loc->second,
           flags::detail::getFlag<flags::label>(std::forward<Flags>(flags)...),
           getColour<Flags...>(), getEmphasis(flags...)});
    return std::move(*this);
  }

  /// Overload of the above but with a 'label' parameter passed to the 'label'
  /// flag for convenience.
  template <class T, class U, class... Flags>
  auto addHighlight(const T& start, const U& end, std::string label,
                    Flags&&... flags)
      -> std::enable_if_t<areLocationProviders<T, U> &&
                              std::conjunction_v<flags::detail::IsFlag<
                                  std::decay_t<Flags>>...> &&
                              isDiagnosticsDocManagerBuilder(),
                          DiagnosticsBuilder&&> {
    return addHighlight(start, end, flags::label = std::move(label),
                        std::forward<Flags>(flags)...);
  }

  /// Overload of the above but with a single positon and a 'label' parameter
  /// passed to the 'label' flag for convenience.
  template <class T, class... Flags>
  auto addHighlight(const T& pos, std::string label, Flags&&... flags)
      -> std::enable_if_t<areLocationProviders<T> &&
                              std::conjunction_v<flags::detail::IsFlag<
                                  std::decay_t<Flags>>...> &&
                              isDiagnosticsDocManagerBuilder(),
                          DiagnosticsBuilder&&> {
    return addHighlight(pos, flags::label = std::move(label),
                        std::forward<Flags>(flags)...);
  }

  /// Adds a new note to the diagnostic. 'location' as well as 'message' and its
  /// 'args' serve the same purpose as in the constructor. The document and
  /// context are taken from the 'DiagnosticsDocManager' it was initially
  /// constructed with.
  template <class T, class S, class... Args>
  auto addNote(const T& location, const S& message, Args&&... args)
      -> std::enable_if_t<areLocationProviders<T> &&
                              isDiagnosticsDocManagerBuilder(),
                          DiagnosticsBuilder&&> {
    Location loc = rangeLoc(location);
    m_diagnostic.messages.push_back(
        {Severity::Note,
         loc ? m_diagnostic.messages.back().document : nullptr,
         loc ? loc->first : 0,
         fmt::format(message, std::forward<Args>(args)...),
         {}});
    return std::move(*this);
  }

  /// Adds a new note to the diagnostic. 'message' and its 'args' serve the same
  /// purpose as in the constructor.
  template <class S, class... Args, class M = Manager>
  auto addNote(const S& message, Args&&... args)
      -> std::enable_if_t<std::is_same_v<M, DiagnosticsNoDocManager>,
                          DiagnosticsBuilder&&> {
    m_diagnostic.messages.push_back(
        {Severity::Note,
         nullptr,
         0,
         fmt::format(message, std::forward<Args>(args)...),
         {}});
    return std::move(*this);
  }
};

template <class Context, class T, class S, class... Args>
DiagnosticsBuilder(DiagnosticsDocManager<Context>&, Severity, const T&,
                   const S&, Args&&...)
    -> DiagnosticsBuilder<DiagnosticsDocManager<Context>>;

template <class T, class S, class... Args>
DiagnosticsBuilder(DiagnosticsDocManager<void>&, Severity, const T&, const S&,
                   Args&&...)
    -> DiagnosticsBuilder<DiagnosticsDocManager<void>>;

template <class T, class S, class... Args>
DiagnosticsBuilder(const Document&, Severity, const T&, const S&, Args&&...)
    -> DiagnosticsBuilder<DiagnosticsDocManager<void>>;

template <class S, class... Args>
DiagnosticsBuilder(DiagnosticsNoDocManager&, Severity, const S&, Args&&...)
    -> DiagnosticsBuilder<DiagnosticsNoDocManager>;

} // namespace pylir::Diag

template <>
struct fmt::formatter<llvm::StringRef> : formatter<string_view> {
  template <class Context>
  auto format(llvm::StringRef string, Context& ctx) const {
    return fmt::formatter<string_view>::format({string.data(), string.size()},
                                               ctx);
  }
};
