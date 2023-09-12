# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(PylirTablegen)

add_library(CodeGenNew CodeGenNew.cpp)
target_link_libraries(CodeGenNew PUBLIC PylirHIRDialect Parser
  PRIVATE PylirTransformsUtils MLIRArithDialect MLIRControlFlowDialect)
