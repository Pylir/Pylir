# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_passes(Passes Transforms PREFIX PylirMem)

add_library(PylirMemTransforms
  HeapToStack.cpp
)
add_dependencies(PylirMemTransforms
  PylirMemTransformsPassIncGen
)
target_link_libraries(PylirMemTransforms
  PUBLIC
  MLIRPass
  
  PRIVATE
  PylirAnalysis
  PylirMemDialect
)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirMemTransforms)
