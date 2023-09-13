# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(Catch)

add_executable(diagnostics_tests
  diagnostics_tests.cpp
  document_tests.cpp
)
target_link_libraries(diagnostics_tests
  Catch2::Catch2WithMain
  Diagnostics
)
catch_discover_tests(diagnostics_tests)
