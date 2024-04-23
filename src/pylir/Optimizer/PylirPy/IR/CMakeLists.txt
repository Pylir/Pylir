# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_dialect(PylirPy py)

add_pylir_interface(ATTR PylirPyAttrInterfaces)
add_pylir_rewriter(PylirPyPatterns)

set(LLVM_TARGET_DEFINITIONS PylirPyAttrInterfaces.td)
pylir_tablegen(PylirPyWrapInterfaces.h.inc -gen-wrap-interfaces)
add_public_tablegen_target(PylirPyWrapInterfacesIncGen)

add_library(PylirPyDialect
  PylirPyAttrInterfaces.cpp
  PylirPyAttributes.cpp
  PylirPyDialect.cpp
  PylirPyOps.cpp
  PylirPyOpFold.cpp
  PylirPyOpSROA.cpp
  PylirPyVerifiers.cpp
  PylirPyTraits.cpp
  PylirPyTypes.cpp
  Value.cpp
)
add_dependencies(PylirPyDialect
  PylirPyIncGen
  PylirPyPatternsIncGen
  PylirPyAttrInterfacesIncGen
  PylirPyWrapInterfacesIncGen
)
target_link_libraries(PylirPyDialect
  PUBLIC
  PylirAttrVerifyInterface
  PylirCaptureInterface
  PylirMemoryFoldInterface
  PylirPyCopyObjectInterface
  PylirPyExceptionHandlingInterface
  PylirPyKnownTypeObjectInterface
  PylirPyOnlyReadsValueInterface
  PylirSROAInterfaces
  PylirSupport
  
  MLIRFunctionInterfaces
  MLIRIR
  MLIRInferTypeOpInterface
  
  PRIVATE
  PylirTransformsUtils
  
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRDataLayoutInterfaces
)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirPyDialect)
