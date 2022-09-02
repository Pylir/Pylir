// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <pylir/Optimizer/PylirPy/Analysis/TypeFlow.hpp>

#include "Passes.hpp"

namespace pylir::test
{
#define GEN_PASS_DEF_TESTTYPEFLOWPASS
#include "Passes.h.inc"
} // namespace pylir::test

namespace
{

class TestTypeFlow : public pylir::test::impl::TestTypeFlowPassBase<TestTypeFlow>
{
protected:
    void runOnOperation() override;

public:
    using Base::Base;
};

void TestTypeFlow::runOnOperation()
{
    for (auto iter : getOperation().getOps<mlir::FunctionOpInterface>())
    {
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
