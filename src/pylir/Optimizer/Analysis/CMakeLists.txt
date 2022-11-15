# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_dialect(MemorySSAIR memSSA)

add_library(PylirAnalysis MemorySSA.cpp MemorySSAIR.cpp InlineCost.cpp AliasSetTracker.cpp LoopInfo.cpp EscapeAnalysis.cpp)
target_link_libraries(PylirAnalysis PUBLIC MLIRPass MLIRAnalysis
  PRIVATE PylirCaptureInterface PRIVATE PylirTransformsUtils)
add_dependencies(PylirAnalysis MemorySSAIRIncGen)
