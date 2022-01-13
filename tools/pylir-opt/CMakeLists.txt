
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Test)
add_public_tablegen_target(TestPassIncGen)

add_executable(pylir-opt main.cpp TestMemorySSA.cpp)
target_link_libraries(pylir-opt PRIVATE MLIRMlirOptMain ${dialect_libs} ${conversion_libs} PylirAnalysis)
target_include_directories(pylir-opt SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(pylir-opt TestPassIncGen)
