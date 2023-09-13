# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(Lexer
  Lexer.cpp
  Token.cpp
)
target_link_libraries(Lexer
  PUBLIC
  PylirSupport
  
  LLVMSupport
  
  PUBLIC
  Diagnostics)
