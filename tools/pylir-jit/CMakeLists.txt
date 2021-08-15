
llvm_map_components_to_libnames(llvm_jit ExecutionEngine OrcJIT RuntimeDyld)

add_executable(pylir-jit main.cpp)
target_link_libraries(pylir-jit PylirMain ${llvm_jit})
