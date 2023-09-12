//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pylir/Runtime/Objects/Objects.hpp>

#include <vector>

namespace pylir::rt {
std::pair<std::uintptr_t, std::uintptr_t>
collectStackRoots(std::vector<PyObject*>& results);
} // namespace pylir::rt
