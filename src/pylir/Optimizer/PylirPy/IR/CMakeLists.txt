# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS PylirPyOps.td)
mlir_tablegen(PylirPyOps.h.inc -gen-op-decls)
mlir_tablegen(PylirPyOps.cpp.inc -gen-op-defs)
pylir_tablegen(PylirPyOpsExtra.cpp.inc -gen-op-variable-decorators)
mlir_tablegen(PylirPyTypes.h.inc -gen-typedef-decls --typedefs-dialect="py")
mlir_tablegen(PylirPyTypes.cpp.inc -gen-typedef-defs --typedefs-dialect="py")
mlir_tablegen(PylirPyDialect.h.inc -gen-dialect-decls -dialect="py")
mlir_tablegen(PylirPyDialect.cpp.inc -gen-dialect-defs -dialect="py")
mlir_tablegen(PylirPyEnums.h.inc -gen-enum-decls)
mlir_tablegen(PylirPyEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(PylirPyAttributes.cpp.inc -gen-attrdef-defs
  --attrdefs-dialect="py")
set(LLVM_TARGET_DEFINITIONS PylirPyAttributes.td)
mlir_tablegen(PylirPyAttributes.h.inc -gen-attrdef-decls
  --attrdefs-dialect="py")
set(LLVM_TARGET_DEFINITIONS PylirPyAttrBase.td)
mlir_tablegen(PylirPyAttrBase.h.inc -gen-attrdef-decls
  --attrdefs-dialect="py")
add_public_tablegen_target(PylirPyIncGen)
add_pylir_doc(PylirPyOps.td PylirPyDialect Dialect/ -gen-dialect-doc
  -dialect="py")

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
  PylirPyOpsVerifier.cpp
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
