# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_rewriter(PylirPyToPylirMem)

add_library(PylirPyToPylirMem PylirPyToPylirMem.cpp)
add_dependencies(PylirPyToPylirMem PylirConversionPassIncGen PylirPyToPylirMemIncGen)
target_link_libraries(PylirPyToPylirMem PUBLIC PylirPyDialect PylirMemDialect MLIRPass MLIRTransforms MLIRArithDialect)
set_property(GLOBAL APPEND PROPERTY MLIR_CONVERSION_LIBS PylirPyToPylirMem)
