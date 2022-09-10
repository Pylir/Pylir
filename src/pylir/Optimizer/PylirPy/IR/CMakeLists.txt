# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_dialect(PylirPy py)
add_pylir_interface(ATTR ObjectAttrInterface)
add_pylir_interface(TYPE ObjectTypeInterface)
add_pylir_interface(ATTR IntAttrInterface)
add_pylir_interface(OP TypeRefineableInterface)
add_pylir_rewriter(PylirPyPatterns)

add_library(PylirPyDialect PylirPyDialect.cpp PylirPyOps.cpp PylirPyAttributes.cpp PylirPyTypes.cpp PylirPyOpFold.cpp
        ObjectAttrInterface.cpp IntAttrInterface.cpp ObjectTypeInterface.cpp TypeRefineableInterface.cpp Value.cpp
        PylirPyOpSROA.cpp)
add_dependencies(PylirPyDialect PylirPyIncGen PylirPyPatternsIncGen ObjectAttrInterfaceIncGen IntAttrInterfaceIncGen
        ObjectTypeInterfaceIncGen TypeRefineableInterfaceIncGen)
target_link_libraries(PylirPyDialect PUBLIC PylirSupport MLIRIR MLIRInferTypeOpInterface PylirMemoryFoldInterface
        PylirCaptureInterface PylirPyExceptionHandlingInterface PylirPyObjectFromTypeObjectInterface
        PylirPyCopyObjectInterface PylirPyOnlyReadsValueInterface PylirSROAInterfaces
        PRIVATE MLIRArithmeticDialect MLIRControlFlowDialect PylirTransformsUtils)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirPyDialect)
if (PYLIR_USE_PCH)
    target_link_libraries(PylirPyDialect PRIVATE COMMON_PCH)
    target_precompile_headers(PylirPyDialect PRIVATE <mlir/IR/Operation.h> <mlir/IR/BuiltinOps.h>)
endif ()

