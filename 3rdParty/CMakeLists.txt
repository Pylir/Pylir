# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(fmt-7.1.3 EXCLUDE_FROM_ALL)
add_subdirectory(utf8proc EXCLUDE_FROM_ALL)
add_subdirectory(libtommath EXCLUDE_FROM_ALL)
if (MSVC)
    set(LIBUNWIND_ENABLE_SHARED OFF)
    add_compile_definitions(__SEH__)
    add_compile_options(-fno-sanitize=all -fno-profile-instr-generate -fno-coverage-mapping)
    add_subdirectory(libunwind EXCLUDE_FROM_ALL)
endif ()
