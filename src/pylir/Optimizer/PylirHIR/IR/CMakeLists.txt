# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_dialect(PylirHIR pyHIR)

add_pylir_interface(OP PylirHIRInterfaces)
add_pylir_interface(OP PylirHIRDerivedInterfaces FILE PylirHIROps)

add_library(PylirHIRDialect
  PylirHIRDialect.cpp
  PylirHIROps.cpp
)
add_dependencies(PylirHIRDialect
  PylirHIRDerivedInterfacesIncGen
  PylirHIRInterfacesIncGen
  PylirHIRIncGen
)
target_link_libraries(PylirHIRDialect
  PUBLIC
  PylirPyDialect
  PylirPyExceptionHandlingInterface
  
  MLIRControlFlowInterfaces
  MLIRIR
)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirHIRDialect)
