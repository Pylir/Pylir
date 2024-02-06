# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_passes(Passes Test NO_DOC)
add_pylir_dialect(Test test NO_DOC)

add_executable(pylir-opt
  main.cpp
  TestAddChange.cpp
  TestAliasSetTracker.cpp
  TestDialect.cpp
  TestEscapeAnalysis.cpp
  TestHelloWorld.cpp
  TestInlineAll.cpp
  TestInlinerInterface.cpp
  TestLinker.cpp
  TestLoopInfo.cpp
  TestMemorySSA.cpp
)
target_link_libraries(pylir-opt
  PRIVATE
  PylirAnalysis
  PylirExternalModels
  PylirHIRDialect
  PylirHIRToPylirPy
  PylirHIRTransforms
  PylirMemDialect
  PylirMemTransforms
  PylirLinker
  PylirOptimizer
  PylirPyToPylirMem
  PylirPyTransforms
  PylirPyTransformsUtil
  PylirToLLVMIR
  PylirTransforms
  
  MLIRArithToLLVM
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIROptLib
  MLIRReconcileUnrealizedCasts
  MLIRTransforms
)
target_include_directories(pylir-opt SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(pylir-opt TestPassIncGen)
add_dependencies(pylir-opt TestIncGen)
