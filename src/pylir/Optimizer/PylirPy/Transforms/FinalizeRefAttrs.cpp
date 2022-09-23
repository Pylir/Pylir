//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/BuiltinOps.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

#include "Passes.hpp"

namespace pylir::Py
{
#define GEN_PASS_DEF_FINALIZEREFATTRSPASS
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"
} // namespace pylir::Py

namespace
{
struct FinalizeRefAttrs : pylir::Py::impl::FinalizeRefAttrsPassBase<FinalizeRefAttrs>
{
protected:
    void runOnOperation() override
    {
        getOperation()->walk([&](pylir::Py::GlobalValueOp valueOp) { pylir::Py::RefAttr::get(valueOp); });
    }
};
} // namespace
