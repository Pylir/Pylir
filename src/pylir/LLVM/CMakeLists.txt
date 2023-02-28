# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(PylirLLVMPasses PylirGC.cpp Upstream/RewriteStatepointsForGC.cpp MarkSanitizersGCLeafs.cpp)
target_link_libraries(PylirLLVMPasses
        PUBLIC
        LLVMCore

        PRIVATE
        LLVMAnalysis
        LLVMCodeGen
        LLVMMC
        LLVMAsmPrinter
        LLVMTarget
        LLVMSupport
        LLVMTransformUtils
        )
