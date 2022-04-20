# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(PylirPyUtil Util.cpp)
target_link_libraries(PylirPyUtil PUBLIC PylirPyDialect)
if (PYLIR_USE_PCH)
    target_link_libraries(PylirPyUtil PRIVATE COMMON_PCH)
    target_precompile_headers(PylirPyUtil PRIVATE <mlir/IR/Operation.h> <mlir/IR/BuiltinOps.h>)
endif ()
