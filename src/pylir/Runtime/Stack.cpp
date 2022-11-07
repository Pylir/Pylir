//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Stack.hpp"

/// The StackMap is a data structured generated by the compiler, making it possible to read all references live at so
/// called state points. By walking the stack, it is then possible to read all live references on the stack in the
/// current stack trace, as long as a call was made through a state point in LLVM.
///
/// Conceptually, the stack map is simply a mapping from Program Counter to a list of alive references. Alive reference
/// locations are encoded the following way in memory:
///
/// struct (packed) ReferenceLocation {
///   enum class Type : std::uint8_t {
///     Register = 1, /// The references is stored within a caller saved register.
///     Direct = 2, /// A object allocated on the stack. Must be traversed to find more alive references.
///     Indirect = 3, /// The reference was spilled onto the stack
///   } type;
///   /// The DWARF register number. Depending on 'type' this may either be the register within the reference is stored
///   /// or either the frame or stack pointer which will be used to read the spilled or allocated on stack object.
///   uleb128 registerNumber;
///   if (type != Register) {
///     /// Offset of where a spilled or allocated on the stack object is located, relative to the frame or stack
///     /// pointer referred to by 'registerNumber'.
///     sleb128 offset;
///   }
///   if (type == Indirect) {
///     /// Amount of references that are allocated contiguously after each other at the address referred to by
///     /// 'registerNumber' and 'offset'.
///     uint8_t count;
///   }
/// };
///
/// Reading a reference of type 'Register' is simply reading that register. Reading a 'Direct' location is simply
/// 'readReg(registerNumber) + offset'. The resulting pointer then refers to the on the stack allocated object.
/// A 'Indirection' location is resolved by computing '*(readReg(registerNumber) + offset)' which will then yield the
/// reference to an object.
///
/// To now map these location structures to program counters they'll now be put into the stack map structures, which is
/// our top level structure:
///
/// struct (packed) Stackmap {
///     uint32_t magic; /// Must contain 0x50594C52, aka the ascii string 'PYLR' interpreted as uint32_t.
///     /// Amount of reference locations.
///     uleb128 referenceLocationCount;
///     /// All reference locations. Each location must be unique.
///     ReferenceLocation locations[referenceLocationCount];
///     /// Amount of callsites within the Stack map.
///     uleb128 callSiteCount;
///
///     struct Callsite {
///         /// Program counter of the call, as found in the stacktrace.
///         uintptr_t programCounter;
///         /// Amount of references within this calls frame that are alive at this call.
///         uleb128 referenceLocationCount;
///         /// Indices within 'locations' of the Stackmap.
///         uleb128 locationIndices[referenceLocationCount];
///     };
///     Callsite callsites[callSiteCount];
/// };
///

#include <cstdlib>
#include <unordered_map>
#include <vector>

#ifdef __linux__
    #define UNW_LOCAL_ONLY
    #include <libunwind.h>
#endif

#include <pylir/Support/Util.hpp>

#include "API.hpp"

namespace
{
/// Magic appearing at the beginning of the stack map as a simple integrity test.
constexpr std::uint32_t PYLR_MAGIC = 0x50594C52;
} // namespace

/// The stack map symbol with the name used by the compiler.
extern "C" const std::uint8_t pylir_stack_map;

namespace
{

/// Runtime deserialized representation of a reference location. See the file synopsis for precise details.
struct ReferenceLocation
{
    enum class Type : std::uint8_t
    {
        Register = 1,
        Direct = 2,
        Indirect = 3,
    } type;
    std::uint8_t count;
    int registerNumber;
    std::uint32_t offset;
};

/// Runtime deserialized representation of our Stack map. See the files synopsis for further details.
class Stackmap
{
    std::vector<ReferenceLocation> m_referenceLocations;
    /// Program counter map to list of indices into 'm_referenceLocations'.
    std::unordered_map<std::uintptr_t, std::vector<std::uint32_t>> m_callSites;

public:
    Stackmap(std::vector<ReferenceLocation>&& referenceLocations,
             std::unordered_map<std::uintptr_t, std::vector<std::uint32_t>>&& callSites)
        : m_referenceLocations(std::move(referenceLocations)), m_callSites(std::move(callSites))
    {
    }

    /// Iterable range of reference locations.
    class ReferenceRange
    {
        const std::vector<ReferenceLocation>* m_referenceLocations;
        const std::vector<std::uint32_t>* m_callSiteRef;

        friend class Stackmap;

        ReferenceRange(const std::vector<ReferenceLocation>* referenceLocations,
                       const std::vector<std::uint32_t>* callSiteRef)
            : m_referenceLocations(referenceLocations), m_callSiteRef(callSiteRef)
        {
        }

    public:
        using value_type = ReferenceLocation;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        /// Iterator over the 'ReferenceRange'.
        class const_iterator
        {
            const std::uint32_t* m_pos{};
            const std::vector<ReferenceLocation>* m_locs{};

            friend class ReferenceRange;

            const_iterator(const std::uint32_t* pos, const std::vector<ReferenceLocation>* locs)
                : m_pos(pos), m_locs(locs)
            {
            }

        public:
            using value_type = ReferenceLocation;
            using difference_type = std::ptrdiff_t;
            using reference = const value_type&;
            using pointer = const value_type*;
            // Not more required for the time being. Can be made a random_access_iterator with a lot more boilerplate.
            using iterator_category = std::forward_iterator_tag;

            const_iterator() = default;

            friend bool operator!=(const_iterator lhs, const_iterator rhs)
            {
                return lhs.m_pos != rhs.m_pos;
            }

            reference operator*() const
            {
                return (*m_locs)[*m_pos];
            }

            pointer operator->() const
            {
                return &**this;
            }

            const_iterator& operator++()
            {
                m_pos++;
                return *this;
            }

            const_iterator operator++(int)
            {
                auto copy = *this;
                ++(*this);
                return copy;
            }
        };

        using iterator = const_iterator;

        /// Returns the begin iterator to the first 'ReferenceLocation'.
        [[nodiscard]] const_iterator begin() const
        {
            return {m_callSiteRef ? m_callSiteRef->data() : nullptr, m_referenceLocations};
        }

        /// Returns the end iterator past the last 'ReferenceLocation'.
        [[nodiscard]] const_iterator end() const
        {
            return {m_callSiteRef ? m_callSiteRef->data() + m_callSiteRef->size() : nullptr, m_referenceLocations};
        }
    };

    /// Returns the range of reference locations for a specific program counter. If the program counter does have any
    /// reference locations associated with it, an empty range is returned instead.
    [[nodiscard]] ReferenceRange getReferencesForPC(std::uintptr_t programCounter) const
    {
        auto res = m_callSites.find(programCounter);
        if (res == m_callSites.end())
        {
            return {&m_referenceLocations, nullptr};
        }
        return {&m_referenceLocations, &res->second};
    }
};

/// Returns the stack map singleton. Lazily parses in the stack map on first use and does so in a thread safe manner as
/// well.
const Stackmap& getStackMap()
{
    static Stackmap stackmap = []
    {
        const std::uint8_t* curr = &pylir_stack_map;
        std::uint32_t magic;
        std::memcpy(&magic, curr, sizeof(std::uint32_t));
        PYLIR_ASSERT(magic == PYLR_MAGIC);
        curr += sizeof(std::uint32_t);

        std::vector<ReferenceLocation> referenceLocations(pylir::rt::readULEB128(&curr));
        for (ReferenceLocation& location : referenceLocations)
        {
            location.type = static_cast<ReferenceLocation::Type>(*curr);
            curr++;
            location.registerNumber = pylir::rt::readULEB128(&curr);
            if (location.type != ReferenceLocation::Type::Register)
            {
                location.offset = pylir::rt::readSLEB128(&curr);
            }
            if (location.type == ReferenceLocation::Type::Indirect)
            {
                location.count = *curr;
                curr++;
            }
        }

        std::size_t callSiteCount = pylir::rt::readULEB128(&curr);
        std::unordered_map<std::uintptr_t, std::vector<std::uint32_t>> callSites(callSiteCount);
        for (std::size_t i = 0; i < callSiteCount; i++)
        {
            curr = pylir::roundUpTo(curr, alignof(std::uintptr_t));
            std::uintptr_t programCounter;
            std::memcpy(&programCounter, curr, sizeof(std::uintptr_t));
            curr += sizeof(std::uintptr_t);

            std::vector<std::uint32_t> indices(pylir::rt::readULEB128(&curr));
            for (std::uint32_t& index : indices)
            {
                index = pylir::rt::readULEB128(&curr);
            }
            callSites.insert_or_assign(programCounter, std::move(indices));
        }
        return Stackmap(std::move(referenceLocations), std::move(callSites));
    }();
    return stackmap;
}

} // namespace

std::pair<std::uintptr_t, std::uintptr_t> pylir::rt::collectStackRoots(std::vector<PyObject*>& results)
{
    std::uintptr_t stackLowerBound = std::numeric_limits<std::uintptr_t>::max();
    std::uintptr_t stackUpperBound = 0;
#ifdef __linux__
    unw_context_t uc;
    unw_getcontext(&uc);
    unw_cursor_t cursor;
    unw_init_local(&cursor, &uc);
    while (unw_step(&cursor) > 0)
    {
        unw_word_t programCounter;
        unw_get_reg(&cursor, UNW_REG_IP, &programCounter);
        for (const auto& iter : getStackMap().getReferencesForPC(programCounter))
        {
            switch (iter.type)
            {
                case ReferenceLocation::Type::Register:
                {
                    unw_word_t rp;
                    unw_get_reg(&cursor, iter.registerNumber, &rp);
                    if (!rp)
                    {
                        break;
                    }
                    results.push_back(reinterpret_cast<pylir::rt::PyObject*>(rp));
                    break;
                }
                case ReferenceLocation::Type::Direct:
                {
                    unw_word_t rp;
                    unw_get_reg(&cursor, iter.registerNumber, &rp);
                    auto* object = reinterpret_cast<pylir::rt::PyObject*>(rp + iter.offset);
                    std::uintptr_t sentinel;
                    std::memcpy(&sentinel, object, sizeof(std::uintptr_t));
                    if (!sentinel)
                    {
                        break;
                    }

                    results.push_back(object);
                    stackLowerBound = std::min(stackLowerBound, reinterpret_cast<std::uintptr_t>(object));
                    stackUpperBound = std::max(stackUpperBound, reinterpret_cast<std::uintptr_t>(object));
                    break;
                }
                case ReferenceLocation::Type::Indirect:
                {
                    unw_word_t rp;
                    unw_get_reg(&cursor, iter.registerNumber, &rp);
                    auto** ptr = reinterpret_cast<pylir::rt::PyObject**>(rp + iter.offset);
                    for (std::size_t i = 0; i < iter.count; i++)
                    {
                        auto* object = ptr[i];
                        if (!object)
                        {
                            continue;
                        }
                        results.push_back(object);
                    }
                    break;
                }
            }
        }
    }
#else
    auto trace = [&](_Unwind_Context* context)
    {
        uintptr_t programCounter = _Unwind_GetIP(context);
        for (const auto& iter : getStackMap().getReferencesForPC(programCounter))
        {
            switch (iter.type)
            {
                case ReferenceLocation::Type::Register:
                {
                    auto* object = reinterpret_cast<pylir::rt::PyObject*>(_Unwind_GetGR(context, iter.registerNumber));
                    if (!object)
                    {
                        break;
                    }
                    results.push_back(object);
                    break;
                }
                case ReferenceLocation::Type::Direct:
                {
                    auto* object = reinterpret_cast<pylir::rt::PyObject*>(_Unwind_GetGR(context, iter.registerNumber)
                                                                          + iter.offset);
                    std::uintptr_t sentinel;
                    std::memcpy(&sentinel, object, sizeof(std::uintptr_t));
                    if (!sentinel)
                    {
                        break;
                    }
                    results.push_back(object);
                    stackLowerBound = std::min(stackLowerBound, reinterpret_cast<std::uintptr_t>(object));
                    stackUpperBound = std::max(stackUpperBound, reinterpret_cast<std::uintptr_t>(object));
                    break;
                }
                case ReferenceLocation::Type::Indirect:
                {
                    auto* ptr = reinterpret_cast<pylir::rt::PyObject**>(_Unwind_GetGR(context, iter.registerNumber)
                                                                        + iter.offset);
                    for (std::size_t i = 0; i < iter.count; i++)
                    {
                        auto* object = ptr[i];
                        if (!object)
                        {
                            continue;
                        }
                        results.push_back(object);
                    }

                    break;
                }
            }
        }
    };
    _Unwind_Backtrace(
        +[](_Unwind_Context* context, void* lambda)
        {
            (*reinterpret_cast<decltype(trace)*>(lambda))(context);
            return _URC_NO_REASON;
        },
        reinterpret_cast<void*>(&trace));
#endif
    return {stackLowerBound, stackUpperBound};
}
