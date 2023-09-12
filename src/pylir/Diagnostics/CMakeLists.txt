# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(Diagnostics
  DiagnosticsBuilder.cpp
  DiagnosticsManager.cpp
  Document.cpp
)
target_link_libraries(Diagnostics
  PRIVATE
  PylirSupport
  
  PUBLIC
  fmt::fmt
  LLVMSupport
)
