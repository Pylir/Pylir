
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_executable(pylir-opt main.cpp)
target_link_libraries(pylir-opt PRIVATE MLIRMlirOptMain ${dialect_libs} ${conversion_libs})
