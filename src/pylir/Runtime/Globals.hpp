// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <tcb/span.hpp>

#include "Objects.hpp"

namespace pylir::rt
{

tcb::span<PyObject**> getHandles();

tcb::span<PyObject*> getCollections();

bool isGlobal(PyObject* object);

} // namespace pylir::rt
