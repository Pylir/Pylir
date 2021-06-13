
add_library(Parser Parser.cpp Dumper.cpp)
target_link_libraries(Parser PUBLIC Lexer)
if (PYLIR_USE_PCH)
    target_link_libraries(Parser PRIVATE COMMON_PCH)
endif ()
