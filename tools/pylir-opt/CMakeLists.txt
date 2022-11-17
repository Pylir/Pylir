# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_pylir_passes(Passes Test)
add_pylir_dialect(Test test)

add_executable(pylir-opt main.cpp TestMemorySSA.cpp TestDialect.cpp TestInlinerInterface.cpp TestTypeFlow.cpp TestAliasSetTracker.cpp TestLoopInfo.cpp TestHelloWorld.cpp TestLinker.cpp TestInlineAll.cpp TestAddChange.cpp TestEscapeAnalysis.cpp)
target_link_libraries(pylir-opt PRIVATE MLIRMlirOptMain ${dialect_libs} ${conversion_libs} PylirAnalysis PylirPyTransformsUtil PylirPyAnalysis PylirLinker PylirOptimizer)
target_include_directories(pylir-opt SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(pylir-opt TestPassIncGen)
add_dependencies(pylir-opt TestIncGen)
