# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS TypeFlowIR.td)
mlir_tablegen(TypeFlowIROps.h.inc -gen-op-decls)
mlir_tablegen(TypeFlowIROps.cpp.inc -gen-op-defs)
mlir_tablegen(TypeFlowIRTypes.h.inc -gen-typedef-decls --typedefs-dialect="typeFlow")
mlir_tablegen(TypeFlowIRTypes.cpp.inc -gen-typedef-defs --typedefs-dialect="typeFlow")
mlir_tablegen(TypeFlowIRAttributes.h.inc -gen-attrdef-decls --attrdefs-dialect="typeFlow")
mlir_tablegen(TypeFlowIRAttributes.cpp.inc -gen-attrdef-defs --attrdefs-dialect="typeFlow")
mlir_tablegen(TypeFlowIRDialect.h.inc -gen-dialect-decls -dialect="typeFlow")
mlir_tablegen(TypeFlowIRDialect.cpp.inc -gen-dialect-defs -dialect="typeFlow")
add_public_tablegen_target(TypeFlowIRIncGen)
add_dependencies(mlir-headers TypeFlowIRIncGen)

set(LLVM_TARGET_DEFINITIONS TypeFlowPasses.td)
mlir_tablegen(TypeFlowPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(TypeFlowPassesIncGen)

add_library(PylirPyAnalysis TypeFlowIR.cpp TypeFlow.cpp)
target_link_libraries(PylirPyAnalysis PUBLIC MLIRIR PRIVATE PylirPyDialect PylirAnalysis MLIRPass)
add_dependencies(PylirPyAnalysis TypeFlowIRIncGen TypeFlowPassesIncGen)
