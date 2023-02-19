# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(PylirToLLVMIR
        PylirToLLVMIR.cpp
        WinX64.cpp
        X86_64.cpp
        PlatformABI.cpp
        PylirTypeConverter.cpp
        CodeGenState.cpp)
add_dependencies(PylirToLLVMIR PylirConversionPassIncGen)
target_link_libraries(PylirToLLVMIR PUBLIC PylirMemDialect MLIRLLVMDialect MLIRPass MLIRTransforms MLIRFuncToLLVM
  MLIRControlFlowToLLVM LLVMTargetParser)
set_property(GLOBAL APPEND PROPERTY MLIR_CONVERSION_LIBS PylirToLLVMIR)
