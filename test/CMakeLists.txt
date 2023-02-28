# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(PYLIR_TEST_DEPENDS
        FileCheck count not split-file
        pylir-opt
        pylir-translate
        pylir
        pylir-tblgen
        pylir-stdlib
        PylirRuntime
        PylirRuntimeMain
        )

# For the 'lit' config, the fuzzer is essentially equal to using ASAN and UBSAN, since that is what non-executable
# source files are compiled with.
if (PYLIR_FUZZER)
    set(PYLIR_SANITIZERS "address,undefined")
endif ()
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        @ONLY
)

add_test(NAME lit-tests COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pylir-lit.py "${CMAKE_CURRENT_BINARY_DIR}" -v)

