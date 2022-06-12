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
    for (auto iter : getOperation().getOps<mlir::FunctionOpInterface>())
    {
        if (iter.isExternal())
        {
            continue;
        }
        auto function = getChildAnalysis<pylir::Py::TypeFlow>(iter).getFunction();
        if (mlir::failed(function.verify()))
        {
            signalPassFailure();
            return;
        }
        function.print(llvm::outs(), mlir::OpPrintingFlags().assumeVerified());
    }
}
} // namespace

std::unique_ptr<mlir::Pass> createTestTypeFlow()
{
    return std::make_unique<TestTypeFlow>();
}
