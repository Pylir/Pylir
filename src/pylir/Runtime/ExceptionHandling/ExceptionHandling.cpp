//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pylir/Runtime/CAPI/API.hpp>
#include <pylir/Runtime/Objects/Objects.hpp>

#include <iostream>

void pylir_raise(pylir::rt::PyBaseException& exception) {
  auto& header = exception.getUnwindHeader();
  std::memset(&header, 0, sizeof(header));
  header.exception_cleanup =
      +[](_Unwind_Reason_Code, _Unwind_Exception*) { /*NOOP for now*/ };
  header.exception_class = pylir::rt::PyBaseException::EXCEPTION_CLASS;
  auto code = _Unwind_RaiseException(&header);
  switch (code) {
  case _URC_END_OF_STACK:
    // TODO call sys.excepthook instead
    extern pylir::rt::PyObject exceptHook asm("sys.__excepthook__");
    exceptHook(type(exception), exception, pylir::rt::Builtins::None);
    std::exit(1);
  case _URC_FATAL_PHASE1_ERROR:
  case _URC_FATAL_PHASE2_ERROR: std::abort();
  default: PYLIR_UNREACHABLE;
  }
}

namespace {

// Heavily borrowed from llvm/examples/ExceptionDemo/ExceptionDemo.cpp

// DWARF Constants
enum {
  // NOLINTBEGIN(readability-identifier-naming): These names are part of a
  // standard.
  DW_EH_PE_absptr = 0x00,
  DW_EH_PE_uleb128 = 0x01,
  DW_EH_PE_udata2 = 0x02,
  DW_EH_PE_udata4 = 0x03,
  DW_EH_PE_udata8 = 0x04,
  DW_EH_PE_sleb128 = 0x09,
  DW_EH_PE_sdata2 = 0x0A,
  DW_EH_PE_sdata4 = 0x0B,
  DW_EH_PE_sdata8 = 0x0C,
  DW_EH_PE_pcrel = 0x10,
  DW_EH_PE_textrel = 0x20,
  DW_EH_PE_datarel = 0x30,
  DW_EH_PE_funcrel = 0x40,
  DW_EH_PE_aligned = 0x50,
  DW_EH_PE_indirect = 0x80,
  DW_EH_PE_omit = 0xFF
  // NOLINTEND(readability-identifier-naming)
};

template <class AsType>
std::uintptr_t readPointerHelper(const std::uint8_t*& p) {
  AsType value;
  std::memcpy(&value, p, sizeof(AsType));
  p += sizeof(AsType);
  return static_cast<std::uintptr_t>(value);
}

std::uintptr_t readEncodedPointer(const std::uint8_t** data,
                                  std::uint8_t encoding) {
  std::uintptr_t result = 0;
  if (encoding == DW_EH_PE_omit)
    return result;

  const std::uint8_t* p = *data;
  // first get value
  switch (encoding & 0x0F) {
  case DW_EH_PE_absptr: result = readPointerHelper<std::uintptr_t>(p); break;
  case DW_EH_PE_uleb128: result = pylir::rt::readULEB128(&p); break;
  case DW_EH_PE_sleb128:
    result = static_cast<std::uintptr_t>(pylir::rt::readSLEB128(&p));
    break;
  case DW_EH_PE_udata2: result = readPointerHelper<std::uint16_t>(p); break;
  case DW_EH_PE_udata4: result = readPointerHelper<std::uint32_t>(p); break;
  case DW_EH_PE_udata8: result = readPointerHelper<std::uint64_t>(p); break;
  case DW_EH_PE_sdata2: result = readPointerHelper<std::int16_t>(p); break;
  case DW_EH_PE_sdata4: result = readPointerHelper<std::int32_t>(p); break;
  case DW_EH_PE_sdata8: result = readPointerHelper<std::int64_t>(p); break;
  default:
    // not supported
    PYLIR_UNREACHABLE;
  }
  // then add relative offset
  switch (encoding & 0x70) {
  case DW_EH_PE_absptr:
    // do nothing
    break;
  case DW_EH_PE_pcrel:
    if (result)
      result += reinterpret_cast<intptr_t>(*data);

    break;
  case DW_EH_PE_datarel:
  case DW_EH_PE_textrel:
  case DW_EH_PE_funcrel:
  case DW_EH_PE_aligned:
  default:
    // not supported
    PYLIR_UNREACHABLE;
  }
  // then apply indirection
  if (result && (encoding & DW_EH_PE_indirect))
    result = *(reinterpret_cast<std::uintptr_t*>(result));

  *data = p;
  return result;
}

pylir::rt::PyTypeObject* readTypeObject(std::uint64_t typeIndex,
                                        const std::uint8_t* classInfo,
                                        std::uint8_t typeEncoding) {
  if (classInfo == nullptr)
    // this should not happen.  Indicates corrupted eh_table.
    PYLIR_UNREACHABLE;

  switch (typeEncoding & 0x0F) {
  case DW_EH_PE_absptr: typeIndex *= sizeof(void*); break;
  case DW_EH_PE_udata2:
  case DW_EH_PE_sdata2: typeIndex *= 2; break;
  case DW_EH_PE_udata4:
  case DW_EH_PE_sdata4: typeIndex *= 4; break;
  case DW_EH_PE_udata8:
  case DW_EH_PE_sdata8: typeIndex *= 8; break;
  default:
    // this should not happen.   Indicates corrupted eh_table.
    PYLIR_UNREACHABLE;
  }
  classInfo -= typeIndex;
  return reinterpret_cast<pylir::rt::PyTypeObject*>(
      readEncodedPointer(&classInfo, typeEncoding));
}

struct Result {
  _Unwind_Reason_Code code;
  std::uintptr_t landingPad;
  std::uint32_t typeIndex;
};

Result findLandingPad(_Unwind_Action actions, bool nativeException,
                      _Unwind_Exception* exception, _Unwind_Context* context) {
  // Inconsistent states
  if (actions & _UA_SEARCH_PHASE) {
    if (actions & (_UA_CLEANUP_PHASE | _UA_HANDLER_FRAME | _UA_FORCE_UNWIND))
      return {_URC_FATAL_PHASE1_ERROR, 0, 0};

  } else if (actions & _UA_CLEANUP_PHASE) {
    if ((actions & (_UA_HANDLER_FRAME)) && (actions & _UA_FORCE_UNWIND))
      return {_URC_FATAL_PHASE2_ERROR, 0, 0};

  } else {
    return {_URC_FATAL_PHASE1_ERROR, 0, 0};
  }
  const auto* exceptionTable = reinterpret_cast<const uint8_t*>(
      _Unwind_GetLanguageSpecificData(context));
  if (!exceptionTable)
    return {_URC_CONTINUE_UNWIND, 0, 0};

  std::uintptr_t ip = _Unwind_GetIP(context) - 1;
  std::uintptr_t functionStart = _Unwind_GetRegionStart(context);
  auto offset = ip - functionStart;
  auto encodingStart = *exceptionTable++;
  const auto* lpStart = reinterpret_cast<const uint8_t*>(
      readEncodedPointer(&exceptionTable, encodingStart));
  if (!lpStart)
    lpStart = reinterpret_cast<const uint8_t*>(functionStart);

  const std::uint8_t* classInfo = nullptr;
  std::uint8_t typeEncoding = *exceptionTable++;
  if (typeEncoding != DW_EH_PE_omit) {
    auto classInfoOffset = pylir::rt::readULEB128(&exceptionTable);
    classInfo = exceptionTable + classInfoOffset;
  }
  std::uint8_t callSiteEncoding = *exceptionTable++;
  auto callSiteTableLength =
      static_cast<std::uint32_t>(pylir::rt::readULEB128(&exceptionTable));
  const auto* callSiteTableStart = exceptionTable;
  const auto* callSiteTableEnd = callSiteTableStart + callSiteTableLength;
  const auto* actionTableStart = callSiteTableEnd;
  for (const auto* callSitePtr = callSiteTableStart;
       callSitePtr < callSiteTableEnd;) {
    auto start = readEncodedPointer(&callSitePtr, callSiteEncoding);
    auto length = readEncodedPointer(&callSitePtr, callSiteEncoding);
    auto landingPad = readEncodedPointer(&callSitePtr, callSiteEncoding);
    auto actionEntry = pylir::rt::readULEB128(&callSitePtr);
    if (offset >= (start + length))
      continue;
    PYLIR_ASSERT(offset >= start);

    if (landingPad == 0)
      return {_URC_CONTINUE_UNWIND, 0, 0};

    landingPad = reinterpret_cast<std::uintptr_t>(lpStart) + landingPad;
    if (actionEntry == 0)
      return {actions & _UA_SEARCH_PHASE ? _URC_CONTINUE_UNWIND
                                         : _URC_HANDLER_FOUND,
              landingPad, 0};

    const auto* action = actionTableStart + (actionEntry - 1);
    bool hasCleanUp = false;
    while (true) {
      // auto* actionRecord = action;
      std::int32_t typeIndex = pylir::rt::readSLEB128(&action);
      if (typeIndex > 0) {
        // catch clauses
        auto* typeObject = readTypeObject(typeIndex, classInfo, typeEncoding);
        if (!nativeException &&
            typeObject == &pylir::rt::Builtins::BaseException) {
          // TODO: synthesize some kind of foreign exception object that can't
          // be handled by user code.
          //       That way we can safely execute code, such as finally blocks.
          //       Those can't be part of cleanup as finally code may actually
          //       stop the exception
          PYLIR_ASSERT(false && "Not yet implemented");
        }
        auto* pyException =
            pylir::rt::PyBaseException::fromUnwindHeader(exception);
        if (isinstance(*pyException, *typeObject))
          return {_URC_HANDLER_FOUND, landingPad,
                  static_cast<std::uint32_t>(typeIndex)};

      } else if (typeIndex < 0) {
        // Don't support filters or anything of the sort at the moment as I
        // don't have a need yet
      } else {
        // cleanup clause
        hasCleanUp = true;
      }
      const auto* temp = action;
      auto actionOffset = pylir::rt::readSLEB128(&temp);
      if (actionOffset == 0)
        return {hasCleanUp && (actions & _UA_CLEANUP_PHASE)
                    ? _URC_HANDLER_FOUND
                    : _URC_CONTINUE_UNWIND,
                landingPad, static_cast<std::uint32_t>(typeIndex)};

      action += actionOffset;
    }
  }
  PYLIR_UNREACHABLE;
}

_Unwind_Reason_Code personalityImpl(int version, _Unwind_Action actions,
                                    std::uint64_t clazz,
                                    _Unwind_Exception* exceptionInfo,
                                    _Unwind_Context* context) {
  if (version != 1 || !exceptionInfo || !context)
    return _URC_FATAL_PHASE1_ERROR;

  bool nativeException = clazz == pylir::rt::PyBaseException::EXCEPTION_CLASS;
  if (actions == (_UA_CLEANUP_PHASE | _UA_HANDLER_FRAME) && nativeException) {
    auto* pyException =
        pylir::rt::PyBaseException::fromUnwindHeader(exceptionInfo);
    _Unwind_SetGR(context, __builtin_eh_return_data_regno(0),
                  reinterpret_cast<uintptr_t>(exceptionInfo));
    _Unwind_SetGR(context, __builtin_eh_return_data_regno(1),
                  static_cast<uintptr_t>(pyException->getTypeIndex()));
    _Unwind_SetIP(context, pyException->getLandingPad());
    return _URC_INSTALL_CONTEXT;
  }
  auto result =
      findLandingPad(actions, nativeException, exceptionInfo, context);
  if (result.code == _URC_CONTINUE_UNWIND ||
      result.code == _URC_FATAL_PHASE1_ERROR)
    return result.code;

  if (actions & _UA_SEARCH_PHASE) {
    PYLIR_ASSERT(result.code == _URC_HANDLER_FOUND);
    if (nativeException) {
      auto* pyException =
          pylir::rt::PyBaseException::fromUnwindHeader(exceptionInfo);
      pyException->setLandingPad(result.landingPad);
      pyException->setTypeIndex(result.typeIndex);
    }
    return _URC_HANDLER_FOUND;
  }

  // TODO: Trying to jump to a catch handler with a foreign exception. Figure
  // out how that is supposed
  //       to work
  _Unwind_SetGR(context, __builtin_eh_return_data_regno(0),
                reinterpret_cast<uintptr_t>(exceptionInfo));
  _Unwind_SetGR(context, __builtin_eh_return_data_regno(1),
                static_cast<uintptr_t>(result.typeIndex));
  _Unwind_SetIP(context, result.landingPad);
  return _URC_INSTALL_CONTEXT;
}

} // namespace

#ifdef _WIN64

#define WIN32_MEAN_AND_LEAN
#include <windows.h>

// NOLINTNEXTLINE(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" EXCEPTION_DISPOSITION
_GCC_specific_handler(EXCEPTION_RECORD* exc, void* frame, CONTEXT* ctx,
                      DISPATCHER_CONTEXT* disp, _Unwind_Personality_Fn pers);

extern "C" EXCEPTION_DISPOSITION
pylir_personality_function(PEXCEPTION_RECORD msExec, void* frame,
                           PCONTEXT msOrigContext, PDISPATCHER_CONTEXT msDisp) {
  return _GCC_specific_handler(msExec, frame, msOrigContext, msDisp,
                               &personalityImpl);
}

#elif !defined(__USING_SJLJ_EXCEPTIONS__)

extern "C" _Unwind_Reason_Code pylir_personality_function(
    int version, _Unwind_Action action, std::uint64_t clazz,
    _Unwind_Exception* exceptionInfo, _Unwind_Context* context) {
  return personalityImpl(version, action, clazz, exceptionInfo, context);
}

#else
#error Not implemented
#endif
