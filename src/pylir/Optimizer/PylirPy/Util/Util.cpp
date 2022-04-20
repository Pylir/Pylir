// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Util.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include "Builtins.hpp"

namespace
{
void implementBlock(mlir::OpBuilder& builder, mlir::Block* block)
{
    PYLIR_ASSERT(block);
    if (auto* next = builder.getBlock()->getNextNode())
    {
        block->insertBefore(next);
    }
    else
    {
        builder.getBlock()->getParent()->push_back(block);
    }
    builder.setInsertionPointToStart(block);
}

void raiseException(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value exception,
                    mlir::Block* PYLIR_NULLABLE exceptionHandler)
{
    if (exceptionHandler)
    {
        builder.create<mlir::cf::BranchOp>(loc, exceptionHandler, exception);
    }
    else
    {
        builder.create<pylir::Py::RaiseOp>(loc, exception);
    }
}

mlir::Value buildCall(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value self, mlir::Value tuple,
                      mlir::Value kwargs, mlir::Block* exceptionHandler)
{
    mlir::Value result;
    if (!exceptionHandler)
    {
        result = builder.create<pylir::Py::CallMethodOp>(loc, self, tuple, kwargs);
    }
    else
    {
        auto* happyPath = new mlir::Block;
        result = builder.create<pylir::Py::CallMethodExOp>(loc, self, tuple, kwargs, mlir::ValueRange{},
                                                           mlir::ValueRange{}, happyPath, exceptionHandler);
        implementBlock(builder, happyPath);
    }
    auto failure = builder.create<pylir::Py::IsUnboundValueOp>(loc, result);
    auto* typeCall = new mlir::Block;
    auto* notBound = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(loc, failure, notBound, typeCall);

    implementBlock(builder, notBound);
    auto typeError = pylir::Py::buildException(loc, builder, pylir::Py::Builtins::TypeError.name, {}, exceptionHandler);
    raiseException(loc, builder, typeError, exceptionHandler);

    implementBlock(builder, typeCall);
    return result;
}

} // namespace

mlir::Value pylir::Py::buildException(mlir::Location loc, mlir::OpBuilder& builder, std::string_view kind,
                                      std::vector<Py::IterArg> args, mlir::Block* exceptionHandler)
{
    auto typeObj = builder.create<Py::ConstantOp>(loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), kind));
    args.emplace(args.begin(), typeObj);
    mlir::Value tuple;
    if (!exceptionHandler
        || std::none_of(args.begin(), args.end(),
                        [](const Py::IterArg& arg) { return std::holds_alternative<Py::IterExpansion>(arg); }))
    {
        tuple = builder.create<Py::MakeTupleOp>(loc, args);
    }
    else
    {
        auto* happyPath = new mlir::Block;
        tuple = builder.create<Py::MakeTupleExOp>(loc, args, happyPath, mlir::ValueRange{}, exceptionHandler,
                                                  mlir::ValueRange{});
        implementBlock(builder, happyPath);
    }
    auto dict = builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(builder.getContext(), {}));
    auto metaType = builder.create<Py::TypeOfOp>(loc, typeObj);
    auto newMethod = builder.create<Py::GetSlotOp>(loc, typeObj, metaType, "__new__");

    auto obj = builder.create<Py::FunctionCallOp>(loc, newMethod, mlir::ValueRange{newMethod, tuple, dict});
    auto objType = builder.create<Py::TypeOfOp>(loc, obj);
    auto context =
        builder.create<Py::ConstantOp>(loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), Builtins::None.name));
    builder.create<Py::SetSlotOp>(loc, obj, objType, "__context__", context);
    auto cause =
        builder.create<Py::ConstantOp>(loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), Builtins::None.name));
    builder.create<Py::SetSlotOp>(loc, obj, objType, "__cause__", cause);
    return obj;
}

mlir::Value pylir::Py::buildTrySpecialMethodCall(mlir::Location loc, mlir::OpBuilder& builder, llvm::Twine methodName,
                                                 mlir::Value tuple, mlir::Value kwargs, mlir::Block* notFoundPath,
                                                 mlir::Block* exceptionHandler)
{
    auto emptyDict = builder.create<Py::ConstantOp>(loc, Py::DictAttr::get(builder.getContext()));
    if (!kwargs)
    {
        kwargs = emptyDict;
    }
    auto element = builder.create<Py::TupleGetItemOp>(loc, tuple, builder.create<mlir::arith::ConstantIndexOp>(loc, 0));
    auto dropped =
        builder.create<Py::TupleDropFrontOp>(loc, builder.create<mlir::arith::ConstantIndexOp>(loc, 1), tuple);
    auto type = builder.create<Py::TypeOfOp>(loc, element);
    auto mroTuple = builder.create<Py::TypeMROOp>(loc, type).getResult();
    auto lookup = builder.create<Py::MROLookupOp>(loc, mroTuple, methodName.str());
    auto* exec = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(loc, lookup.getSuccess(), exec, notFoundPath);

    implementBlock(builder, exec);
    auto function = builder.create<Py::ConstantOp>(
        loc, mlir::FlatSymbolRefAttr::get(builder.getContext(), Py::Builtins::Function.name));
    auto callableType = builder.create<Py::TypeOfOp>(loc, lookup.getResult());
    auto isFunction = builder.create<Py::IsOp>(loc, callableType, function);
    auto* isFunctionBlock = new mlir::Block;
    auto* notFunctionBlock = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(loc, isFunction, isFunctionBlock, notFunctionBlock);

    implementBlock(builder, isFunctionBlock);
    mlir::Value result;
    if (!exceptionHandler)
    {
        result = builder.create<Py::FunctionCallOp>(loc, lookup.getResult(),
                                                    mlir::ValueRange{lookup.getResult(), tuple, kwargs});
    }
    else
    {
        auto* happyPath = new mlir::Block;
        result = builder.create<Py::FunctionInvokeOp>(
            loc, lookup.getResult(), mlir::ValueRange{lookup.getResult(), tuple, kwargs}, mlir::ValueRange{},
            mlir::ValueRange{}, happyPath, exceptionHandler);
        implementBlock(builder, happyPath);
    }
    auto* exitBlock = new mlir::Block;
    exitBlock->addArgument(builder.getType<Py::DynamicType>(), loc);
    builder.create<mlir::cf::BranchOp>(loc, exitBlock, result);

    implementBlock(builder, notFunctionBlock);
    mroTuple = builder.create<Py::TypeMROOp>(loc, callableType);
    auto getMethod = builder.create<Py::MROLookupOp>(loc, mroTuple, "__get__");
    auto* isDescriptor = new mlir::Block;
    auto* mergeBlock = new mlir::Block;
    mergeBlock->addArgument(builder.getType<Py::DynamicType>(), loc);
    builder.create<mlir::cf::CondBranchOp>(loc, getMethod.getSuccess(), isDescriptor, mergeBlock,
                                           mlir::ValueRange{lookup.getResult()});

    implementBlock(builder, isDescriptor);
    auto selfType = builder.create<Py::TypeOfOp>(loc, element);
    result = buildCall(loc, builder, getMethod.getResult(),
                       builder.create<Py::MakeTupleOp>(loc, std::vector<Py::IterArg>{element, selfType}), emptyDict,
                       exceptionHandler);
    builder.create<mlir::cf::BranchOp>(loc, mergeBlock, result);

    implementBlock(builder, mergeBlock);
    result = buildCall(loc, builder, mergeBlock->getArgument(0), dropped, kwargs, exceptionHandler);
    builder.create<mlir::cf::BranchOp>(loc, exitBlock, result);

    implementBlock(builder, exitBlock);
    return exitBlock->getArgument(0);
}

mlir::Value pylir::Py::buildSpecialMethodCall(mlir::Location loc, mlir::OpBuilder& builder, llvm::Twine methodName,
                                              mlir::Value tuple, mlir::Value kwargs, mlir::Block* exceptionHandler)
{
    auto* notFound = new mlir::Block;
    auto result = buildTrySpecialMethodCall(loc, builder, methodName, tuple, kwargs, notFound, exceptionHandler);
    mlir::OpBuilder::InsertionGuard guard{builder};
    implementBlock(builder, notFound);
    auto exception = Py::buildException(loc, builder, Py::Builtins::TypeError.name, {}, exceptionHandler);
    raiseException(loc, builder, exception, exceptionHandler);
    return result;
}
