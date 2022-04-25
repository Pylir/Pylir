# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Test)
add_public_tablegen_target(TestPassIncGen)

set(LLVM_TARGET_DEFINITIONS TestDialect.td)
mlir_tablegen(TestDialectOps.h.inc -gen-op-decls)
mlir_tablegen(TestDialectOps.cpp.inc -gen-op-defs)
mlir_tablegen(TestDialectTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TestDialectTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(TestDialectAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(TestDialectAttributes.cpp.inc -gen-attrdef-defs)
mlir_tablegen(TestDialect.h.inc -gen-dialect-decls -dialect="test")
mlir_tablegen(TestDialect.cpp.inc -gen-dialect-defs -dialect="test")
add_public_tablegen_target(TestDialectIncGen)
add_dependencies(mlir-headers TestDialectIncGen)

add_executable(pylir-opt main.cpp TestMemorySSA.cpp TestDialect.cpp TestInlinerInterface.cpp TestTypeFlow.cpp)
target_link_libraries(pylir-opt PRIVATE MLIRMlirOptMain ${dialect_libs} ${conversion_libs} PylirAnalysis PylirPyTransformsUtil PylirPyAnalysis)
target_include_directories(pylir-opt SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(pylir-opt TestPassIncGen)
add_dependencies(pylir-opt TestDialectIncGen)
