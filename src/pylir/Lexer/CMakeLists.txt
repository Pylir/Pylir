
add_library(Lexer Lexer.cpp Token.cpp)
target_link_libraries(Lexer PUBLIC LLVMSupport Support PRIVATE fmt::fmt)
