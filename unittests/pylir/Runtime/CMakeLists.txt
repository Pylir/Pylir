# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(PylirCompile)

pylir_obj_compile(builtins.o empty.py)
add_library(PylirTestRuntime STATIC builtins.o)
target_link_libraries(PylirTestRuntime PUBLIC PylirRuntime)

add_subdirectory(MarkAndSweep)
