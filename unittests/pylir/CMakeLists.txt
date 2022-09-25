# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if (MSVC AND ("${PYLIR_SANITIZER}" MATCHES ".*address.*"))
  add_compile_definitions(CATCH_CONFIG_NO_WINDOWS_SEH)
endif ()

add_subdirectory(Diagnostics)
add_subdirectory(Lexer)
add_subdirectory(Parser)
add_subdirectory(Support)
add_subdirectory(Runtime)
