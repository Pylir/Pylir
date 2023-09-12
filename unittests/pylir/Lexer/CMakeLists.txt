# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(Catch)

add_executable(lexer_tests
  lexer_tests.cpp
)
target_link_libraries(lexer_tests
  Catch2::Catch2WithMain
  Lexer
)
catch_discover_tests(lexer_tests)
