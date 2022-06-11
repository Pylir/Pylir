// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Passes.hpp"

#include <pylir/Optimizer/Analysis/LoopInfo.hpp>

#include <llvm/ADT/PostOrderIterator.h>

namespace
{
class TestLoopInfo : public TestLoopInfoBase<TestLoopInfo>
{
public:
protected:
    void runOnOperation() override
    {
        for (auto iter : getOperation().getOps<mlir::FunctionOpInterface>())
        {
            getChildAnalysis<pylir::LoopInfo>(iter).print(llvm::outs());
        }
    }
};
} // namespace

std::unique_ptr<mlir::Pass> createTestLoopInfo()
{
    return std::make_unique<TestLoopInfo>();
}
