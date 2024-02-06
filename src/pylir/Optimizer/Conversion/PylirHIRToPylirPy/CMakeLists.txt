# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(PylirHIRToPylirPy
  PylirHIRToPylirPy.cpp
)
add_dependencies(PylirHIRToPylirPy
  PylirConversionPassIncGen
)
target_link_libraries(PylirHIRToPylirPy
  PRIVATE
  PylirHIRDialect
  PylirPyDialect

  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRPass
  MLIRTransforms
)
