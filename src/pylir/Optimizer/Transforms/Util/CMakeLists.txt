# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(PylirTransformsUtils SSABuilder.cpp SSAUpdater.cpp BranchOpInterfacePatterns.cpp)
target_link_libraries(PylirTransformsUtils PUBLIC MLIRIR MLIRControlFlowInterfaces MLIRSideEffectInterfaces)
