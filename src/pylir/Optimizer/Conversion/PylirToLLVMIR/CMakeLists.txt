# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(PylirToLLVMIR
  CodeGenState.cpp
  PlatformABI.cpp
  PylirToLLVMIR.cpp
  PylirTypeConverter.cpp
  WinX64.cpp
  X86_64.cpp
)
add_dependencies(PylirToLLVMIR
  PylirConversionPassIncGen
)
target_link_libraries(PylirToLLVMIR
  PRIVATE
  LLVMTargetParser
  PylirMemDialect
  
  MLIRControlFlowToLLVM
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
)
set_property(GLOBAL APPEND PROPERTY MLIR_CONVERSION_LIBS PylirToLLVMIR)
