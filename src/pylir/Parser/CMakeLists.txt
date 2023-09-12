# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(Parser
  Dumper.cpp
  Parser.cpp
  ParserCompounds.cpp
  ParserExpressions.cpp
  ParserStatements.cpp
  SemanticAnalysis.cpp
  Syntax.cpp
  Visitor.cpp
)
target_link_libraries(Parser
  PUBLIC
  Lexer
)
