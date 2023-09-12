# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(PylirCompile)

pylir_obj_compile(TARGET builtins.o SOURCE empty.py)
add_library(PylirTestRuntime STATIC builtins.o)
target_link_libraries(PylirTestRuntime
  PUBLIC
  PylirRuntime
)

add_subdirectory(MarkAndSweep)

include(CheckTypeSize)

# stackmap_source.ll.in is relatively target independent, except that it
# "makes use of" size_t. We configure and and substitute the integer width of it
# here.
check_type_size(size_t SIZE_T_BYTES)
math(EXPR SIZE_T_BITS "${SIZE_T_BYTES} * 8")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/stackmap_source.ll.in
  ${CMAKE_CURRENT_BINARY_DIR}/stackmap_source.ll
  @ONLY
)

# '-mllvm -pylir-emit-stackmap=0' disables the generation of all the stack map
# related variables. Necessary for a few tests that create their own stackmap.
pylir_obj_compile(TARGET empty.o SOURCE empty.py
  FLAGS -mllvm -pylir-emit-stackmap=0)
add_library(PylirTestNoStackMapRuntime STATIC empty.o)
target_link_libraries(PylirTestNoStackMapRuntime PUBLIC PylirRuntime)

pylir_obj_compile(TARGET stackmap_source.o
  SOURCE ${CMAKE_CURRENT_BINARY_DIR}/stackmap_source.ll FLAGS -O3)
add_executable(stackmap_reader
  stackmap_reader.cpp
  stackmap_source.o
)
target_link_libraries(stackmap_reader
  PRIVATE
  Catch2::Catch2WithMain
  
  PylirMarkAndSweep
  PylirTestNoStackMapRuntime
)
catch_discover_tests(stackmap_reader)
