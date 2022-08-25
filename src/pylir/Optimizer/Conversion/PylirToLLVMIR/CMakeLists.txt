# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_conversion_library(PylirToLLVMIR
        PylirToLLVMIR.cpp
        WinX64.cpp
        X86_64.cpp
        ARM64.cpp
        PlatformABI.cpp

        DEPENDS
        PylirConversionPassIncGen

        LINK_LIBS PUBLIC
        PylirMemDialect
        MLIRLLVMDialect
        MLIRPass
        MLIRTransforms
        MLIRFuncToLLVM
        MLIRControlFlowToLLVM
        )

if (PYLIR_USE_PCH)
    target_link_libraries(PylirToLLVMIR PRIVATE COMMON_PCH)
    target_precompile_headers(PylirToLLVMIR PRIVATE <mlir/IR/Operation.h> <mlir/IR/BuiltinOps.h>)
endif ()
