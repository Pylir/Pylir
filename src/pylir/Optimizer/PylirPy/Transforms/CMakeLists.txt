# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transform)
add_public_tablegen_target(PylirPyTransformPassIncGen)

add_library(PylirPyTransforms ExpandPyDialect.cpp FoldHandles.cpp HandleLoadStoreElimination.cpp Monomorph.cpp Inliner.cpp TrialInlining.cpp Monomorph.cpp)
add_dependencies(PylirPyTransforms PylirPyTransformPassIncGen)
target_link_libraries(PylirPyTransforms
        PUBLIC
        MLIRPass
        PRIVATE
        PylirAnalysis
        PylirPyAnalysis
        PylirPyDialect
        PylirPyUtil
        PylirTransformsUtils
        PylirPyTransformsUtil
        MLIRTransforms
        MLIRAnalysis
        MLIRFuncDialect
        )
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirPyTransforms)

add_subdirectory(Util)
