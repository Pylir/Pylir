// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/InliningUtils.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Util/InlinerUtil.hpp>

#include <memory>

namespace pylir::MemSSA
{
class MemorySSADialect;
}

namespace
{
#define GEN_PASS_CLASSES
#include "Passes.h.inc"

class TestInlinerInterface : public TestInlinerInterfaceBase<TestInlinerInterface>
{
protected:
    void runOnOperation() override
    {
        llvm::SmallVector<mlir::CallOpInterface> calls;
        getOperation()->walk([&](mlir::CallOpInterface call) { calls.push_back(call); });
        mlir::SymbolTableCollection collection;
        for (auto iter : calls)
        {
            auto ref =
                iter.getCallableForCallee().dyn_cast<mlir::SymbolRefAttr>().dyn_cast_or_null<mlir::FlatSymbolRefAttr>();
            if (!ref || !ref.getValue().startswith("inline"))
            {
                continue;
            }
            auto func = mlir::dyn_cast_or_null<mlir::CallableOpInterface>(iter.resolveCallable(&collection));
            if (!func)
            {
                iter->emitError("Could not resolve function") << ref;
                signalPassFailure();
                return;
            }
            if (mlir::failed(pylir::Py::inlineCall(iter, func)))
            {
                iter->emitError("Inlining ") << ref << " failed";
                signalPassFailure();
                return;
            }
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createTestInlinerInterface()
{
    return std::make_unique<TestInlinerInterface>();
}
