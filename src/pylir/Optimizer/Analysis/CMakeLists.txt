# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_dialect(MemorySSAIR memSSA)

add_library(PylirAnalysis
  AliasSetTracker.cpp
  EscapeAnalysis.cpp
  InlineCost.cpp
  LoopInfo.cpp
  MemorySSA.cpp
  MemorySSAIR.cpp
)
target_link_libraries(PylirAnalysis
  PUBLIC
  MLIRAnalysis
  MLIRPass
  
  PRIVATE
  PylirCaptureInterface
  PylirTransformsUtils
)
add_dependencies(PylirAnalysis MemorySSAIRIncGen)
