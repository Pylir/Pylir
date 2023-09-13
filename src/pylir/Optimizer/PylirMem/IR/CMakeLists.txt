# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_dialect(PylirMem pyMem)

add_library(PylirMemDialect
  PylirMemDialect.cpp
  PylirMemOps.cpp
  Value.cpp
)
add_dependencies(PylirMemDialect
  PylirMemIncGen
)
target_link_libraries(PylirMemDialect
  PUBLIC
  PylirPyDialect
  PylirSupport
  
  MLIRIR
  MLIRInferTypeOpInterface
)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirMemDialect)
