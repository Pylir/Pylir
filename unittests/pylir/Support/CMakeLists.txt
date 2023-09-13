# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(Catch)

add_executable(support_tests
  bigint_tests.cpp
  hashtable_tests.cpp
  text_tests.cpp
  util_tests.cpp
)
target_link_libraries(support_tests
  Catch2::Catch2WithMain
  PylirSupport
)
catch_discover_tests(support_tests)
