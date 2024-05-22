# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_passes(Passes Transform PREFIX PylirHIR)

add_library(PylirHIRTransforms
  ClassBodyOutlining.cpp
  FuncOutlining.cpp
)
add_dependencies(PylirHIRTransforms
  PylirHIRTransformPassIncGen
)
target_link_libraries(PylirHIRTransforms
  PUBLIC
  MLIRPass

  PRIVATE
  PylirHIRDialect
  PylirPyTransformsUtil

  MLIRArithDialect
  MLIRTransformUtils
)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirHIRTransforms)
