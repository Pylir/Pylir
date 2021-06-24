
include(Catch)

add_executable(lexer_tests lexer_tests.cpp main.cpp)
target_link_libraries(lexer_tests Lexer)
catch_discover_tests(lexer_tests)

add_executable(lexer_test_runner test_runner.cpp)
target_link_libraries(lexer_test_runner Lexer)
