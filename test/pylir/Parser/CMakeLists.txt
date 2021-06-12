
include(Catch)

add_executable(parser_tests parser_tests.cpp main.cpp)
target_link_libraries(parser_tests Parser)
catch_discover_tests(parser_tests)
