// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/AsmState.h>
#include <mlir/Pass/Pass.h>

#include <pylir/Optimizer/Analysis/AliasSetTracker.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyTraits.hpp>

#include "Passes.hpp"

namespace
{
class TestAliasSetTracker : public TestAliasSetTrackerBase<TestAliasSetTracker>
{
protected:
    void runOnOperation() override
    {
        for (auto iter : getOperation().getOps<mlir::FunctionOpInterface>())
        {
            auto& aliasAnalysis = getChildAnalysis<mlir::AliasAnalysis>(iter);
            pylir::AliasSetTracker tracker(aliasAnalysis);
            iter.walk(
                [&](mlir::Operation* op)
                {
                    if (op == &op->getBlock()->front())
                    {
                        for (auto& iter2 : op->getBlock()->getArguments())
                        {
                            if (!iter2.getType().isa<pylir::Py::DynamicType>())
                            {
                                continue;
                            }
                            tracker.insert(iter2);
                        }
                    }
                    if (op->hasTrait<pylir::Py::ReturnsImmutable>())
                    {
                        return;
                    }
                    for (auto res : op->getResults())
                    {
                        if (!res.getType().isa<pylir::Py::DynamicType>())
                        {
                            continue;
                        }
                        tracker.insert(res);
                    }
                });

            auto state = mlir::AsmState(iter);
            llvm::outs() << "Alias sets for " << iter.getName() << ":\n";
            for (auto& iter2 : tracker)
            {
                llvm::outs() << "{";
                for (auto iter3 : iter2)
                {
                    iter3.printAsOperand(llvm::outs(), state);
                    llvm::outs() << " ";
                }
                llvm::outs() << "}\n";
            }
        }
    }
};
} // namespace

std::unique_ptr<mlir::Pass> createTestAliasSetTracker()
{
    return std::make_unique<TestAliasSetTracker>();
}
