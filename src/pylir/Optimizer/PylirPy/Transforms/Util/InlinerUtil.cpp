// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "InlinerUtil.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Transforms/InliningUtils.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

mlir::LogicalResult pylir::Py::inlineCall(mlir::CallOpInterface call, mlir::CallableOpInterface callable)
{
    auto invoke = mlir::dyn_cast<pylir::Py::InvokeOp>(*call);
    mlir::Block* after = nullptr;
    if (invoke)
    {
        after = invoke->getBlock()->getNextNode();
    }
    mlir::InlinerInterface interface(call.getContext());
    if (mlir::failed(mlir::inlineCall(interface, call, callable, callable.getCallableRegion())))
    {
        return mlir::failure();
    }
    if (invoke)
    {
        mlir::Block* destBlock;
        if (!after)
        {
            destBlock = &invoke->getParentRegion()->back();
        }
        else
        {
            destBlock = after->getPrevNode();
        }
        auto builder = mlir::OpBuilder::atBlockEnd(destBlock);
        builder.create<mlir::cf::BranchOp>(invoke.getLoc(), invoke.getHappyPath(), invoke.getNormalDestOperands());
    }
    call.erase();
    return mlir::success();
}
