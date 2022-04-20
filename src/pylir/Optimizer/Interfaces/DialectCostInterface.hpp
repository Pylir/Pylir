// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/DialectInterface.h>

namespace pylir
{
class DialectCostInterface : public mlir::DialectInterface::Base<DialectCostInterface>
{
public:
    DialectCostInterface(mlir::Dialect* dialect) : Base(dialect) {}

    virtual std::size_t getCost(mlir::Operation* op) const = 0;
};
} // namespace pylir
