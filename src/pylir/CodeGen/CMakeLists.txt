# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(CodeGen CodeGen.cpp CodeGen.hpp BuiltinsImpl.cpp CompilerBuiltins.cpp)
target_link_libraries(CodeGen PUBLIC PylirPyDialect Parser PRIVATE PylirPyUtil PRIVATE PylirTransformsUtils MLIRFuncDialect)
