
add_library(CodeGen CodeGen.cpp CodeGen.hpp Builtins.cpp)
target_link_libraries(CodeGen PylirPyDialect Parser)
