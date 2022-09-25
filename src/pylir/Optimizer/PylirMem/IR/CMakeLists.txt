# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_dialect(PylirMem pyMem)

add_library(PylirMemDialect PylirMemDialect.cpp PylirMemOps.cpp)
add_dependencies(PylirMemDialect PylirMemIncGen IntAttrInterfaceIncGen)
target_link_libraries(PylirMemDialect PUBLIC MLIRIR PylirPyDialect PylirSupport MLIRInferTypeOpInterface)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirMemDialect)

if (PYLIR_USE_PCH)
  target_link_libraries(PylirMemDialect PRIVATE COMMON_PCH)
  target_precompile_headers(PylirMemDialect PRIVATE <mlir/IR/Operation.h> <mlir/IR/BuiltinOps.h>)
endif ()
