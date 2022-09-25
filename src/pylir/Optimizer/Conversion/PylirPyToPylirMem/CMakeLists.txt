# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_rewriter(PylirPyToPylirMem)

add_library(PylirPyToPylirMem PylirPyToPylirMem.cpp)
add_dependencies(PylirPyToPylirMem PylirConversionPassIncGen PylirPyToPylirMemIncGen)
target_link_libraries(PylirPyToPylirMem PUBLIC PylirPyDialect PylirMemDialect MLIRPass MLIRTransforms)
set_property(GLOBAL APPEND PROPERTY MLIR_CONVERSION_LIBS PylirPyToPylirMem)

if (PYLIR_USE_PCH)
  target_link_libraries(PylirPyToPylirMem PRIVATE COMMON_PCH)
  target_precompile_headers(PylirPyToPylirMem PRIVATE <mlir/IR/Operation.h> <mlir/IR/BuiltinOps.h>)
endif ()
