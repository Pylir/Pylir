
add_library(CodeGen CodeGen.cpp CodeGen.hpp)
target_link_libraries(CodeGen PylirDialect Parser)
