//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

#include <pylir/Support/Util.hpp>

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
        int regNumber;
        std::intptr_t offset;
    };
};

extern "C" const StackMap pylir_default_stack_map = {0x50594C52, 0};

extern "C" const StackMap PYLIR_WEAK_VAR(pylir_stack_map, pylir_default_stack_map);

namespace
{

const std::unordered_map<std::uintptr_t, std::vector<StackMap::Location>>& counterToLoc()
{
    static std::unordered_map<std::uintptr_t, std::vector<StackMap::Location>> result = []
    {
        const auto& debug = pylir_stack_map;
        PYLIR_ASSERT(debug.magic == 0x50594C52);
        const auto* curr = reinterpret_cast<const std::uint8_t*>(&debug + 1);
        std::unordered_map<std::uintptr_t, std::vector<StackMap::Location>> result(debug.callSiteCount);
        for (std::size_t i = 0; i < debug.callSiteCount; i++)
        {
            curr = pylir::roundUpTo(curr, alignof(void*));

            std::uintptr_t programCounter;
            std::memcpy(&programCounter, curr, sizeof(std::uintptr_t));
            curr += sizeof(std::uintptr_t);
            auto locationCount = pylir::rt::readULEB128(&curr);
            auto& vec = result.emplace(programCounter, locationCount).first->second;
            for (std::size_t j = 0; j < locationCount; j++)
            {
                auto type = static_cast<StackMap::Location::Type>(*curr++);
                int regNumber = pylir::rt::readULEB128(&curr);
                std::intptr_t offset = 0;
                switch (type)
                {
                    case StackMap::Location::Type::Direct:
                    case StackMap::Location::Type::Indirect: offset = pylir::rt::readSLEB128(&curr);
                    default: break;
                }
                vec[j] = StackMap::Location{type, regNumber, offset};
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
