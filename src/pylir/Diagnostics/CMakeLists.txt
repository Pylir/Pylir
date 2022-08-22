# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(Diagnostics DiagnosticsBuilder.cpp Document.cpp DiagnosticsManager.cpp DiagnosticsManager.hpp)
target_link_libraries(Diagnostics PRIVATE PylirSupport PUBLIC fmt MLIRSupport)
if (PYLIR_USE_PCH)
    target_link_libraries(Diagnostics PRIVATE COMMON_PCH)
    target_precompile_headers(Diagnostics PRIVATE <fmt/format.h>)
endif ()
