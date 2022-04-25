// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Pass/Pass.h>

#include <pylir/Optimizer/PylirPy/Analysis/TypeFlow.hpp>

#include "Passes.hpp"

namespace
{

class TestTypeFlow : public TestTypeFlowBase<TestTypeFlow>
{
protected:
    void runOnOperation() override;
};

void TestTypeFlow::runOnOperation()
{
    llvm::outs() << getAnalysis<pylir::Py::TypeFlow>().getFunction();
}
} // namespace

std::unique_ptr<mlir::Pass> createTestTypeFlow()
{
    return std::make_unique<TestTypeFlow>();
}
