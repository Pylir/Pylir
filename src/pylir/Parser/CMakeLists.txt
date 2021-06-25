
add_library(Parser ParserExpressions.cpp Dumper.cpp ParserStatements.cpp Syntax.cpp ParserCompounds.cpp)
target_link_libraries(Parser PUBLIC Lexer)
if (PYLIR_USE_PCH)
    target_link_libraries(Parser PRIVATE COMMON_PCH)
endif ()
