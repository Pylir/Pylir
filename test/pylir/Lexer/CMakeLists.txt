
include(Catch)

add_executable(lexer_tests lexer_tests.cpp main.cpp)
target_link_libraries(lexer_tests Lexer Catch2::Catch2)
catch_discover_tests(lexer_tests)
