
add_executable(pylir-opt main.cpp)
target_link_libraries(pylir-opt PylirDialect MLIRMlirOptMain PylirToLLVM)
