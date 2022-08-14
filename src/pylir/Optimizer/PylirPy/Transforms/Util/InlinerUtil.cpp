// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "InlinerUtil.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/BlockAndValueMapping.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

namespace
{
void remapInlinedLocations(llvm::iterator_range<mlir::Region::iterator> range, mlir::Location callerLoc)
{
    llvm::DenseMap<mlir::Location, mlir::Location> mappedLocations;
    auto remapOpLoc = [&](mlir::Operation* op)
    {
        auto [iter, inserted] = mappedLocations.try_emplace(op->getLoc(), op->getLoc());
        if (inserted)
        {
            iter->second = mlir::CallSiteLoc::get(op->getLoc(), callerLoc);
        }
        op->setLoc(iter->second);
    };
    for (auto& block : range)
    {
        block.walk(remapOpLoc);
    }
}
} // namespace

void pylir::Py::inlineCall(mlir::CallOpInterface call, mlir::CallableOpInterface callable)
{
    auto exceptionHandler = mlir::dyn_cast<pylir::Py::ExceptionHandlingInterface>(*call);

    auto* preBlock = call->getBlock();
    auto* postBlock = preBlock->splitBlock(call);
    postBlock->addArguments(
        callable.getCallableResults(),
        llvm::to_vector(llvm::map_range(call->getResults(), [](mlir::Value arg) { return arg.getLoc(); })));

    mlir::BlockAndValueMapping mapping;
    mapping.map(callable.getCallableRegion()->getArguments(), call.getArgOperands());
    callable.getCallableRegion()->cloneInto(preBlock->getParent(), postBlock->getIterator(), mapping);

    auto* firstInlinedBlock = preBlock->getNextNode();
    auto inlineBlocksRange = llvm::make_range(firstInlinedBlock->getIterator(), postBlock->getIterator());
    for (auto& iter : inlineBlocksRange)
    {
        if (exceptionHandler)
        {
            for (auto op : llvm::make_early_inc_range(iter.getOps<pylir::Py::AddableExceptionHandlingInterface>()))
            {
                auto* successBlock = iter.splitBlock(mlir::Block::iterator{op});
                auto builder = mlir::OpBuilder::atBlockEnd(&iter);
                auto* newOp = op.cloneWithExceptionHandling(
                    builder, successBlock, exceptionHandler.getExceptionPath(),
                    static_cast<mlir::OperandRange>(exceptionHandler.getUnwindDestOperandsMutable()));
                op->replaceAllUsesWith(newOp);
                op.erase();
                break;
            }
        }
        auto* terminator = iter.getTerminator();
        if (auto raise = mlir::dyn_cast<pylir::Py::RaiseOp>(terminator); raise && exceptionHandler)
        {
            mlir::OpBuilder builder(raise);
            auto ops =
                llvm::to_vector(static_cast<mlir::OperandRange>(exceptionHandler.getUnwindDestOperandsMutable()));
            ops.insert(ops.begin(), raise.getException());
            builder.create<mlir::cf::BranchOp>(raise.getLoc(), exceptionHandler.getExceptionPath(), ops);
            raise.erase();
            continue;
        }
        if (terminator->hasTrait<mlir::OpTrait::ReturnLike>())
        {
            mlir::OpBuilder builder(terminator);
            builder.create<mlir::cf::BranchOp>(terminator->getLoc(), postBlock, terminator->getOperands());
            terminator->erase();
            continue;
        }
    }

    remapInlinedLocations(inlineBlocksRange, call->getLoc());

    preBlock->getOperations().splice(preBlock->end(), firstInlinedBlock->getOperations());
    firstInlinedBlock->erase();
    for (auto [res, arg] : llvm::zip(call->getResults(), postBlock->getArguments()))
    {
        res.replaceAllUsesWith(arg);
    }
    if (exceptionHandler)
    {
        mlir::OpBuilder builder(call);
        builder.create<mlir::cf::BranchOp>(
            call->getLoc(), exceptionHandler.getHappyPath(),
            static_cast<mlir::OperandRange>(exceptionHandler.getNormalDestOperandsMutable()));
    }
    call.erase();
}
