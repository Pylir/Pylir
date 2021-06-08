
add_library(Parser Parser.cpp)
target_link_libraries(Parser PUBLIC Lexer)
