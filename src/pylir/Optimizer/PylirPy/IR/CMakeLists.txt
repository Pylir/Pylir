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
mlir_tablegen(PylirPyAttributes.cpp.inc -gen-attrdef-defs --attrdefs-dialect="py")
set(LLVM_TARGET_DEFINITIONS PylirPyAttributes.td)
mlir_tablegen(PylirPyAttributes.h.inc -gen-attrdef-decls --attrdefs-dialect="py")
set(LLVM_TARGET_DEFINITIONS PylirPyRefAttr.td)
mlir_tablegen(PylirPyAttrBase.h.inc -gen-attrdef-decls --attrdefs-dialect="py")
add_public_tablegen_target(PylirPyIncGen)
add_pylir_doc(PylirPyOps.td PylirPyDialect Dialect/ -gen-dialect-doc -dialect="py")

add_pylir_interface(ATTR ObjectAttrInterface)
add_pylir_rewriter(PylirPyPatterns)

add_library(PylirPyDialect PylirPyDialect.cpp PylirPyOps.cpp PylirPyAttributes.cpp PylirPyTypes.cpp PylirPyOpFold.cpp
  ObjectAttrInterface.cpp Value.cpp
  PylirPyOpSROA.cpp PylirPyOpsVerifier.cpp PylirPyTraits.cpp)
add_dependencies(PylirPyDialect PylirPyIncGen PylirPyPatternsIncGen ObjectAttrInterfaceIncGen)
target_link_libraries(PylirPyDialect PUBLIC PylirSupport MLIRIR MLIRInferTypeOpInterface PylirMemoryFoldInterface
  PylirCaptureInterface PylirPyExceptionHandlingInterface PylirPyKnownTypeObjectInterface
  PylirPyCopyObjectInterface PylirPyOnlyReadsValueInterface PylirSROAInterfaces
  PRIVATE MLIRArithDialect MLIRControlFlowDialect PylirTransformsUtils MLIRDataLayoutInterfaces)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirPyDialect)
