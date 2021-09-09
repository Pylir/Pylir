
add_library(CodeGen CodeGen.cpp CodeGen.hpp GlobalDiscoverer.cpp GlobalDiscoverer.hpp)
target_link_libraries(CodeGen PylirPyDialect Parser)
