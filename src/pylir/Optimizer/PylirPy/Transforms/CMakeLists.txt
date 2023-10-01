# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_passes(Passes Transform PREFIX PylirPy)

add_library(PylirPyTransforms
  ExpandPyDialect.cpp
  FoldGlobals.cpp
  GlobalLoadStoreElimination.cpp
  GlobalSROA.cpp
  Inliner.cpp
)
add_dependencies(PylirPyTransforms
  PylirPyTransformPassIncGen
)
target_link_libraries(PylirPyTransforms
  PUBLIC
  MLIRPass
  
  PRIVATE
  PylirAnalysis
  PylirPyDialect
  PylirPyTransformsUtil
  PylirTransformsUtils
  
  MLIRAnalysis
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRTransforms
)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirPyTransforms)

add_subdirectory(Util)
