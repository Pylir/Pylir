# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS PylirPyOps.td)
mlir_tablegen(PylirPyOps.h.inc -gen-op-decls)
mlir_tablegen(PylirPyOps.cpp.inc -gen-op-defs)
mlir_tablegen(PylirPyOpsTypes.h.inc -gen-typedef-decls --typedefs-dialect="py")
mlir_tablegen(PylirPyOpsTypes.cpp.inc -gen-typedef-defs --typedefs-dialect="py")
mlir_tablegen(PylirPyOpsAttributes.h.inc -gen-attrdef-decls --attrdefs-dialect="py")
mlir_tablegen(PylirPyOpsAttributes.cpp.inc -gen-attrdef-defs --attrdefs-dialect="py")
mlir_tablegen(PylirPyOpsDialect.h.inc -gen-dialect-decls -dialect="py")
mlir_tablegen(PylirPyOpsDialect.cpp.inc -gen-dialect-defs -dialect="py")
mlir_tablegen(PylirPyOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(PylirPyOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(PylirPyOpsIncGen)
add_dependencies(mlir-headers PylirPyOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ObjectAttrInterface.td)
mlir_tablegen(ObjectAttrInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(ObjectAttrInterface.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(ObjectAttrInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS ObjectTypeInterface.td)
mlir_tablegen(ObjectTypeInterface.h.inc -gen-type-interface-decls)
mlir_tablegen(ObjectTypeInterface.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(ObjectTypeInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS IntAttrInterface.td)
mlir_tablegen(IntAttrInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(IntAttrInterface.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(IntAttrInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS PylirPyPatterns.td)
mlir_tablegen(PylirPyPatterns.cpp.inc -gen-rewriters)
add_public_tablegen_target(PylirPyPatternsIncGen)

set(LLVM_TARGET_DEFINITIONS TypeRefineableInterface.td)
mlir_tablegen(TypeRefineableInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(TypeRefineableInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(TypeRefineableInterfaceIncGen)

add_library(PylirPyDialect PylirPyDialect.cpp PylirPyOps.cpp PylirPyAttributes.cpp PylirPyTypes.cpp PylirPyOpFold.cpp
        ObjectAttrInterface.cpp IntAttrInterface.cpp ObjectTypeInterface.cpp TypeRefineableInterface.cpp)
add_dependencies(PylirPyDialect PylirPyOpsIncGen PylirPyPatternsIncGen ObjectAttrInterfaceIncGen IntAttrInterfaceIncGen
        ObjectTypeInterfaceIncGen TypeRefineableInterfaceIncGen)
target_link_libraries(PylirPyDialect PUBLIC PylirSupport MLIRIR MLIRInferTypeOpInterface PylirMemoryFoldInterface
        PylirCaptureInterface PylirPyExceptionHandlingInterface PylirPyObjectFromTypeObjectInterface
        PRIVATE MLIRArithmeticDialect MLIRControlFlowDialect)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirPyDialect)
if (PYLIR_USE_PCH)
    target_link_libraries(PylirPyDialect PRIVATE COMMON_PCH)
    target_precompile_headers(PylirPyDialect PRIVATE <mlir/IR/Operation.h> <mlir/IR/BuiltinOps.h>)
endif ()

