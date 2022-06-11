# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS MemorySSAIR.td)
mlir_tablegen(MemorySSAIROps.h.inc -gen-op-decls)
mlir_tablegen(MemorySSAIROps.cpp.inc -gen-op-defs)
mlir_tablegen(MemorySSAIRTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MemorySSAIRTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(MemorySSAIRAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(MemorySSAIRAttributes.cpp.inc -gen-attrdef-defs)
mlir_tablegen(MemorySSAIRDialect.h.inc -gen-dialect-decls -dialect="memSSA")
mlir_tablegen(MemorySSAIRDialect.cpp.inc -gen-dialect-defs -dialect="memSSA")
add_public_tablegen_target(MemorySSAIRIncGen)
add_dependencies(mlir-headers MemorySSAIRIncGen)

add_library(PylirAnalysis MemorySSA.cpp MemorySSAIR.cpp BodySize.cpp BodySize.hpp AliasSetTracker.cpp AliasSetTracker.hpp LoopInfo.cpp LoopInfo.hpp)
target_link_libraries(PylirAnalysis PUBLIC MLIRPass MLIRAnalysis
        PRIVATE PylirCaptureInterface PRIVATE PylirTransformsUtils)
add_dependencies(PylirAnalysis MemorySSAIRIncGen)
