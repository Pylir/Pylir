# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(Analysis)
add_subdirectory(Conversion)
add_subdirectory(ExternalModels)
add_subdirectory(Interfaces)
add_subdirectory(Linker)
add_subdirectory(PylirHIR)
add_subdirectory(PylirMem)
add_subdirectory(PylirPy)
add_subdirectory(Transforms)

add_library(PylirOptimizer
  Optimizer.cpp
)
target_link_libraries(PylirOptimizer
  PRIVATE
  PylirHIRToPylirPy
  PylirHIRTransforms
  PylirMemTransforms
  PylirPyToPylirMem
  PylirPyTransforms
  PylirTransforms
  PylirToLLVMIR
  
  MLIRArithToLLVM
  MLIRLLVMIRTransforms
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRTransforms
)
