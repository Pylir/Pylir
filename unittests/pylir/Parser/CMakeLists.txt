
include(Catch)

add_executable(parser_tests parser_tests.cpp main.cpp)
target_link_libraries(parser_tests Parser)
catch_discover_tests(parser_tests)

add_executable(parser_test_runner test_runner.cpp)
target_link_libraries(parser_test_runner Parser)
