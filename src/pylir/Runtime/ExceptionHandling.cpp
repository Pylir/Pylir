
#include "API.hpp"

void pylir_raise(pylir::rt::PyBaseException* exception)
{
    auto& header = exception->getUnwindHeader();
    std::memset(&header, 0, sizeof(header));
    header.exception_cleanup = +[](_Unwind_Reason_Code, _Unwind_Exception*) { /*NOOP for now*/ };
    header.exception_class = pylir::rt::PyBaseException::EXCEPTION_CLASS;
    auto code = _Unwind_RaiseException(&header);
    switch (code)
    {
        case _URC_END_OF_STACK:
            // TODO call sys.excepthook
            std::exit(1);
        case _URC_FATAL_PHASE1_ERROR:
        case _URC_FATAL_PHASE2_ERROR: std::abort();
        default: PYLIR_UNREACHABLE;
    }
}

namespace
{
_Unwind_Reason_Code personalityImpl(int version, _Unwind_Action action, _Unwind_Exception_Class clazz,
                                    _Unwind_Exception* exceptionInfo, _Unwind_Context* context)
{
    if (version != 1 || !exceptionInfo || !context)
    {
        return _URC_FATAL_PHASE1_ERROR;
    }
    bool nativeException = clazz == pylir::rt::PyBaseException::EXCEPTION_CLASS;
    if (action == _UA_SEARCH_PHASE)
    {
        // TODO
        return _URC_CONTINUE_UNWIND;
    }
    return _URC_CONTINUE_UNWIND;
}

} // namespace

#ifdef _WIN64

    #define WIN32_MEAN_AND_LEAN
    #include <windows.h>

extern "C" EXCEPTION_DISPOSITION _GCC_specific_handler(EXCEPTION_RECORD* exc, void* frame, CONTEXT* ctx,
                                                       DISPATCHER_CONTEXT* disp, _Unwind_Personality_Fn pers);

extern "C" EXCEPTION_DISPOSITION pylir_personality_function(PEXCEPTION_RECORD msExec, void* frame,
                                                            PCONTEXT msOrigContext, PDISPATCHER_CONTEXT msDisp)
{
    return _GCC_specific_handler(msExec, frame, msOrigContext, msDisp, &personalityImpl);
}

#elif !defined(__USING_SJLJ_EXCEPTIONS__)

extern "C" _Unwind_Reason_Code pylir_personality_function(int version, _Unwind_Action action,
                                                          _Unwind_Exception_Class clazz,
                                                          _Unwind_Exception* exceptionInfo, _Unwind_Context* context)
{
    return personalityImpl(version, action, clazz, exceptionInfo, context);
}

#else
    #error Not implemented
#endif
