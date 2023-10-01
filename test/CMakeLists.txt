# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(PYLIR_TEST_EXEC_FLAGS "" CACHE STRING
  "pylir flags to use when compiling the E2E execution tests")

set(PYLIR_TEST_DEPENDS
  FileCheck count not split-file
  pylir-opt
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
set(PYLIR_LIT_SITE_CFG_OUT ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${PYLIR_LIT_SITE_CFG_OUT}
  @ONLY
)

set(PYLIR_LIT_CFG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
cmake_path(RELATIVE_PATH PYLIR_LIT_SITE_CFG_OUT BASE_DIRECTORY ${PYLIR_TOOLS_DIR})
cmake_path(RELATIVE_PATH PYLIR_LIT_CFG BASE_DIRECTORY ${PYLIR_TOOLS_DIR})
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/pylir-lit.py.in
  ${PYLIR_TOOLS_DIR}/pylir-lit.py
  @ONLY
)

add_custom_target(pylir-test-depends ALL DEPENDS ${PYLIR_TEST_DEPENDS})

add_test(NAME lit-tests COMMAND ${Python3_EXECUTABLE} ${PYLIR_TOOLS_DIR}/pylir-lit.py "${CMAKE_CURRENT_BINARY_DIR}" -v)
set_tests_properties(lit-tests PROPERTIES TIMEOUT 3600)
