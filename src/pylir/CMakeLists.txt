# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(CodeGen)
add_subdirectory(Diagnostics)
add_subdirectory(Lexer)
add_subdirectory(LLVM)
add_subdirectory(Main)
add_subdirectory(Optimizer)
add_subdirectory(Parser)
add_subdirectory(Support)

# Must come after Support due to checking for target existence.
add_subdirectory(Runtime)
