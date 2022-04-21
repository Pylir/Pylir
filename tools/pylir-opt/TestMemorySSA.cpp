// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Analysis/MemorySSA.hpp>

#include <memory>

namespace pylir::Py
{
class PylirPyDialect;
}

namespace
{
#define GEN_PASS_CLASSES
#include "Passes.h.inc"

class TestMemorySSA : public TestMemorySSABase<TestMemorySSA>
{
protected:
    void runOnOperation() override;
};

void TestMemorySSA::runOnOperation()
{
    for (auto func : getOperation().getOps<mlir::FunctionOpInterface>())
    {
        llvm::outs() << getChildAnalysis<pylir::MemorySSA>(func);
    }
}
} // namespace

std::unique_ptr<mlir::Pass> createTestMemorySSA()
{
    return std::make_unique<TestMemorySSA>();
}
