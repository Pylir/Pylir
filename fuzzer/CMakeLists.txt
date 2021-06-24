
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer")

add_executable(lexer_fuzzer lexer_fuzzer.cpp)
target_link_libraries(lexer_fuzzer Lexer)
