# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer")

add_executable(lexer_fuzzer lexer_fuzzer.cpp)
target_link_libraries(lexer_fuzzer Lexer)

add_executable(parser_fuzzer parser_fuzzer.cpp)
target_link_libraries(parser_fuzzer Parser)
