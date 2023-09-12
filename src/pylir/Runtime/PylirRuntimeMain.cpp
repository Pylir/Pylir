//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOLINTNEXTLINE(bugprone-reserved-identifier)
extern "C" void __init__();

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static LONG handler(EXCEPTION_POINTERS* exception_data) {
#define GCC_MAGIC (('G' << 16) | ('C' << 8) | 'C' | (1U << 29))
  if ((exception_data->ExceptionRecord->ExceptionCode & 0x20ffffff) ==
      GCC_MAGIC)
    if ((exception_data->ExceptionRecord->ExceptionFlags &
         EXCEPTION_NONCONTINUABLE) == 0)
      return EXCEPTION_CONTINUE_EXECUTION;

  return EXCEPTION_CONTINUE_SEARCH;
}
#endif

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
#ifdef _MSC_VER
  SetUnhandledExceptionFilter(handler);
#endif
  __init__();
}
