# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(Util)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transform)
add_public_tablegen_target(PylirTransformPassIncGen)

add_library(PylirTransforms LoadForwardingPass.cpp)
add_dependencies(PylirTransforms PylirTransformPassIncGen)
target_link_libraries(PylirTransforms PUBLIC MLIRPass PRIVATE PylirAnalysis PylirMemoryFoldInterface)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirTransforms)
