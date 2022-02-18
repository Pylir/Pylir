#include "Stack.hpp"

#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <tcb/span.hpp>

#ifdef __linux__
    #define UNW_LOCAL_ONLY
    #include <libunwind.h>
#endif

#include "API.hpp"

struct StackMap
{
    // should be 0x50594C52 'PYLR'
    std::uint32_t magic;
    std::uint32_t callSiteCount;
    struct Location
    {
        enum class Type : std::uint8_t
        {
            Register = 1,
            Direct = 2,
            Indirect = 3,
        } type;
        std::uint16_t regNumber;
        std::uint32_t offset;
    };
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    struct CallSite
    {
        std::uintptr_t programCounter;
        std::uint32_t locationCount;
        alignas(Location) char trailing[];
    };
    alignas(CallSite) char trailing[];
#pragma GCC diagnostic pop
};

extern "C" const StackMap pylir_default_stack_map = {0x50594C52, 0};

#ifdef _MSC_VER
extern "C" const StackMap pylir_stack_map;
    #pragma comment(linker, "/alternatename:pylir_stack_map=pylir_default_stack_map")
#else
extern "C" const StackMap pylir_stack_map __attribute__((weak, alias("pylir_default_stack_map")));
#endif

namespace
{

const std::unordered_map<std::uintptr_t, tcb::span<const StackMap::Location>>& counterToLoc()
{
    static std::unordered_map<std::uintptr_t, tcb::span<const StackMap::Location>> result = []
    {
        std::unordered_map<std::uintptr_t, tcb::span<const StackMap::Location>> result;
        const auto& debug = pylir_stack_map;
        PYLIR_ASSERT(debug.magic == 0x50594C52);
        const char* curr = debug.trailing;
        for (std::size_t i = 0; i < debug.callSiteCount; i++)
        {
            const auto* callSite = reinterpret_cast<const StackMap::CallSite*>(curr);
            auto programCounter = callSite->programCounter;
            auto locationCount = callSite->locationCount;
            curr = callSite->trailing;
            result.emplace(std::piecewise_construct, std::forward_as_tuple(programCounter),
                           std::forward_as_tuple(reinterpret_cast<const StackMap::Location*>(curr), locationCount));
            curr += sizeof(StackMap::Location) * locationCount;
            if (auto align = reinterpret_cast<std::uintptr_t>(curr) % alignof(StackMap::CallSite))
            {
                curr += alignof(StackMap::CallSite) - align;
            }
        }
        return result;
    }();
    return result;
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
        auto result = counterToLoc().find(programCounter);
        if (result == counterToLoc().end())
        {
            continue;
        }
        for (const auto& iter : result->second)
        {
            switch (iter.type)
            {
                case StackMap::Location::Type::Register:
                {
                    unw_word_t rp;
                    unw_get_reg(&cursor, iter.regNumber, &rp);
                    if (!rp)
                    {
                        break;
                    }
                    results.push_back(reinterpret_cast<pylir::rt::PyObject*>(rp));
                    break;
                }
                case StackMap::Location::Type::Direct:
                {
                    unw_word_t rp;
                    unw_get_reg(&cursor, iter.regNumber, &rp);
                    auto* object = reinterpret_cast<pylir::rt::PyObject*>(rp + iter.offset);
                    results.push_back(object);
                    stackLowerBound = std::min(stackLowerBound, reinterpret_cast<std::uintptr_t>(object));
                    stackUpperBound = std::max(stackUpperBound, reinterpret_cast<std::uintptr_t>(object));
                    break;
                }
                case StackMap::Location::Type::Indirect:
                {
                    unw_word_t rp;
                    unw_get_reg(&cursor, iter.regNumber, &rp);
                    auto* object = *reinterpret_cast<pylir::rt::PyObject**>(rp + iter.offset);
                    if (!object)
                    {
                        break;
                    }
                    results.push_back(object);
                    break;
                }
            }
        }
    }
#else
    auto trace = [&](_Unwind_Context* context)
    {
        uintptr_t programCounter = _Unwind_GetIP(context);
        auto result = counterToLoc().find(programCounter);
        if (result == counterToLoc().end())
        {
            return;
        }
        for (const auto& iter : result->second)
        {
            switch (iter.type)
            {
                case StackMap::Location::Type::Register:
                {
                    auto* object = reinterpret_cast<pylir::rt::PyObject*>(_Unwind_GetGR(context, iter.regNumber));
                    if (!object)
                    {
                        break;
                    }
                    results.push_back(object);
                    break;
                }
                case StackMap::Location::Type::Direct:
                {
                    auto* object =
                        reinterpret_cast<pylir::rt::PyObject*>(_Unwind_GetGR(context, iter.regNumber) + iter.offset);
                    results.push_back(object);
                    stackLowerBound = std::min(stackLowerBound, reinterpret_cast<std::uintptr_t>(object));
                    stackUpperBound = std::max(stackUpperBound, reinterpret_cast<std::uintptr_t>(object));
                    break;
                }
                case StackMap::Location::Type::Indirect:
                {
                    auto* object =
                        *reinterpret_cast<pylir::rt::PyObject**>(_Unwind_GetGR(context, iter.regNumber) + iter.offset);
                    if (!object)
                    {
                        break;
                    }
                    results.push_back(object);
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
