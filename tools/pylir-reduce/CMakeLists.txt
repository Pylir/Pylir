# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_executable(pylir-reduce EXCLUDE_FROM_ALL
  main.cpp
)
target_link_libraries(pylir-reduce
  PRIVATE
  MLIRReduceLib
  ${dialect_libs}
  ${conversion_libs}
)
