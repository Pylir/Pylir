
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_executable(pylir-reduce main.cpp)
target_link_libraries(pylir-reduce PRIVATE MLIRReduceLib ${dialect_libs} ${conversion_libs})
