//  Copyright 2022 Markus Böck
//
//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>

#include "CodeGen.hpp"

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

mlir::Value buildTrySpecialMethodCall(pylir::PyBuilder& builder, llvm::Twine methodName, mlir::Value args,
                                      mlir::Value kws, mlir::Block* notFoundPath,
                                      mlir::Block* callIntrException = nullptr)
{
    auto element = builder.createTupleGetItem(args, builder.create<mlir::arith::ConstantIndexOp>(0));
    auto elementType = builder.createTypeOf(element);
    auto mroTuple = builder.createTypeMRO(elementType);
    auto lookup = builder.createMROLookup(mroTuple, methodName.str());
    auto failure = builder.createIsUnboundValue(lookup);
    auto* exec = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(builder.getCurrentLoc(), failure, notFoundPath, exec);

    implementBlock(builder, exec);
    mroTuple = builder.createTypeMRO(builder.createTypeOf(lookup.getResult()));
    auto getMethod = builder.createMROLookup(mroTuple, "__get__");
    failure = builder.createIsUnboundValue(getMethod);
    auto* isDescriptor = new mlir::Block;
    auto* mergeBlock = new mlir::Block;
    mergeBlock->addArgument(builder.getDynamicType(), builder.getCurrentLoc());
    builder.create<mlir::cf::CondBranchOp>(builder.getCurrentLoc(), failure, mergeBlock,
                                           mlir::ValueRange{lookup.getResult()}, isDescriptor, mlir::ValueRange{});

    implementBlock(builder, isDescriptor);
    auto tuple = builder.createMakeTuple({element, elementType}, nullptr);
    auto result =
        builder.createPylirCallIntrinsic(getMethod.getResult(), tuple, builder.createMakeDict(), callIntrException);
    builder.create<mlir::cf::BranchOp>(builder.getCurrentLoc(), mergeBlock, result);

    implementBlock(builder, mergeBlock);
    // TODO: This is incorrect. One should be passing all but args[0], as args[0] will already be bound by the __get__
    //       descriptor of function. We haven't yet implemented this however, hence this is the stop gap solution.
    return builder.createPylirCallIntrinsic(mergeBlock->getArgument(0), args, kws, callIntrException);
}

mlir::Value buildSpecialMethodCall(pylir::PyBuilder& builder, llvm::Twine methodName, mlir::Value args, mlir::Value kws,
                                   mlir::Block* callIntrException = nullptr)
{
    auto* notFound = new mlir::Block;
    auto result = buildTrySpecialMethodCall(builder, methodName, args, kws, notFound, callIntrException);
    mlir::OpBuilder::InsertionGuard guard{builder};
    implementBlock(builder, notFound);
    auto exception =
        pylir::buildException(builder.getCurrentLoc(), builder, pylir::Builtins::TypeError.name, {}, nullptr);
    builder.createRaise(exception);
    return result;
}

mlir::Value binOp(pylir::PyBuilder& builder, llvm::StringRef method, llvm::StringRef revMethod, mlir::Value lhs,
                  mlir::Value rhs)
{
    auto trueC = builder.create<mlir::arith::ConstantIntOp>(true, 1);
    auto falseC = builder.create<mlir::arith::ConstantIntOp>(false, 1);
    auto* endBlock = new mlir::Block;
    endBlock->addArgument(builder.getDynamicType(), builder.getCurrentLoc());
    if (method == "__eq__" || method == "__ne__")
    {
        auto isSame = builder.createIs(lhs, rhs);
        auto* continueNormal = new mlir::Block;
        builder.create<mlir::cf::CondBranchOp>(isSame, endBlock,
                                               mlir::ValueRange{builder.createConstant(method == "__eq__")},
                                               continueNormal, mlir::ValueRange{});
        implementBlock(builder, continueNormal);
    }
    auto lhsType = builder.createTypeOf(lhs);
    auto rhsType = builder.createTypeOf(rhs);
    auto sameType = builder.createIs(lhsType, rhsType);
    auto* normalMethodBlock = new mlir::Block;
    normalMethodBlock->addArgument(builder.getI1Type(), builder.getCurrentLoc());
    auto* differentTypeBlock = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(sameType, normalMethodBlock, mlir::ValueRange{trueC}, differentTypeBlock,
                                           mlir::ValueRange{});

    implementBlock(builder, differentTypeBlock);
    auto mro = builder.createTypeMRO(rhsType);
    auto subclass = builder.createTupleContains(mro, lhsType);
    auto* isSubclassBlock = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(subclass, isSubclassBlock, normalMethodBlock, mlir::ValueRange{falseC});

    implementBlock(builder, isSubclassBlock);
    auto rhsMroTuple = builder.createTypeMRO(rhsType);
    auto lookup = builder.createMROLookup(rhsMroTuple, revMethod);
    auto failure = builder.createIsUnboundValue(lookup);
    auto* hasReversedBlock = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(failure, normalMethodBlock, mlir::ValueRange{falseC}, hasReversedBlock,
                                           mlir::ValueRange{});

    implementBlock(builder, hasReversedBlock);
    auto lhsMroTuple = builder.createTypeMRO(lhsType);
    auto lhsLookup = builder.createMROLookup(lhsMroTuple, revMethod);
    failure = builder.createIsUnboundValue(lhsLookup);
    auto* callReversedBlock = new mlir::Block;
    auto* lhsHasReversedBlock = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(failure, callReversedBlock, lhsHasReversedBlock);

    implementBlock(builder, lhsHasReversedBlock);
    auto sameImplementation = builder.createIs(lookup.getResult(), lhsLookup.getResult());
    builder.create<mlir::cf::CondBranchOp>(sameImplementation, normalMethodBlock, mlir::ValueRange{falseC},
                                           callReversedBlock, mlir::ValueRange{});

    implementBlock(builder, callReversedBlock);
    auto tuple = builder.createMakeTuple({rhs, lhs}, nullptr);
    auto dict = builder.createMakeDict();
    auto reverseResult = buildSpecialMethodCall(builder, revMethod, tuple, dict);
    auto isNotImplemented = builder.createIs(reverseResult, builder.createNotImplementedRef());
    builder.create<mlir::cf::CondBranchOp>(isNotImplemented, normalMethodBlock, mlir::ValueRange{trueC}, endBlock,
                                           mlir::ValueRange{reverseResult});

    implementBlock(builder, normalMethodBlock);
    auto* typeErrorBlock = new mlir::Block;
    tuple = builder.createMakeTuple({lhs, rhs}, nullptr);
    dict = builder.createMakeDict();
    auto result = buildTrySpecialMethodCall(builder, method, tuple, dict, typeErrorBlock);
    isNotImplemented = builder.createIs(result, builder.createNotImplementedRef());
    auto* maybeTryReverse = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(isNotImplemented, maybeTryReverse, endBlock, mlir::ValueRange{result});

    implementBlock(builder, maybeTryReverse);
    auto* actuallyTryReverse = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(normalMethodBlock->getArgument(0), typeErrorBlock, actuallyTryReverse);

    implementBlock(builder, actuallyTryReverse);
    tuple = builder.createMakeTuple({rhs, lhs}, nullptr);
    reverseResult = buildTrySpecialMethodCall(builder, revMethod, tuple, dict, typeErrorBlock);
    isNotImplemented = builder.createIs(reverseResult, builder.createNotImplementedRef());
    builder.create<mlir::cf::CondBranchOp>(isNotImplemented, typeErrorBlock, endBlock, mlir::ValueRange{reverseResult});

    implementBlock(builder, typeErrorBlock);
    if (method != "__eq__" && method != "__ne__")
    {
        auto typeError =
            pylir::buildException(builder.getCurrentLoc(), builder, pylir::Builtins::TypeError.name, {}, nullptr);
        builder.createRaise(typeError);
    }
    else
    {
        mlir::Value isEqual = builder.createIs(lhs, rhs);
        if (method == "__ne__")
        {
            isEqual = builder.create<mlir::arith::XOrIOp>(isEqual, trueC);
        }
        mlir::Value boolean = builder.createBoolFromI1(isEqual);
        builder.create<mlir::cf::BranchOp>(endBlock, boolean);
    }

    implementBlock(builder, endBlock);
    return endBlock->getArgument(0);
}

void buildRevBinOpCompilerBuiltin(pylir::PyBuilder& builder, llvm::StringRef functionName, llvm::StringRef method,
                                  llvm::StringRef revMethod)
{
    auto func = builder.create<mlir::func::FuncOp>(
        functionName,
        builder.getFunctionType({builder.getDynamicType(), builder.getDynamicType()}, builder.getDynamicType()));
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(func.addEntryBlock());
    auto result = binOp(builder, method, revMethod, func.getArgument(0), func.getArgument(1));
    builder.create<mlir::func::ReturnOp>(result);
}

void buildBinOpCompilerBuiltin(pylir::PyBuilder& builder, llvm::StringRef functionName, llvm::StringRef method)
{
    auto func = builder.create<mlir::func::FuncOp>(
        functionName,
        builder.getFunctionType({builder.getDynamicType(), builder.getDynamicType()}, builder.getDynamicType()));
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(func.addEntryBlock());
    mlir::Value lhs = func.getArgument(0);
    mlir::Value rhs = func.getArgument(1);
    auto tuple = builder.createMakeTuple({lhs, rhs}, nullptr);
    auto dict = builder.createMakeDict();
    auto result = buildSpecialMethodCall(builder, method, tuple, dict);
    builder.create<mlir::func::ReturnOp>(result);
}

void buildUnaryOpCompilerBuiltin(pylir::PyBuilder& builder, llvm::StringRef functionName, llvm::StringRef method)
{
    auto func = builder.create<mlir::func::FuncOp>(
        functionName, builder.getFunctionType({builder.getDynamicType()}, builder.getDynamicType()));
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(func.addEntryBlock());
    auto tuple = builder.createMakeTuple({func.getArgument(0)}, nullptr);
    auto dict = builder.createMakeDict();
    auto result = buildSpecialMethodCall(builder, method, tuple, dict);
    builder.create<mlir::func::ReturnOp>(result);
}

void buildCallOpCompilerBuiltin(pylir::PyBuilder& builder, llvm::StringRef functionName, llvm::StringRef)
{
    auto func = builder.create<mlir::func::FuncOp>(
        functionName,
        builder.getFunctionType({builder.getDynamicType(), builder.getDynamicType(), builder.getDynamicType()},
                                builder.getDynamicType()));
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(func.addEntryBlock());

    auto self = func.getArgument(0);
    auto args = func.getArgument(1);
    auto kws = func.getArgument(2);

    auto selfType = builder.createTypeOf(self);
    // We have to somehow break this recursion by detecting a function type and calling it directly.
    auto isFunction = builder.createIs(selfType, builder.createFunctionRef());
    auto* isFunctionBlock = new mlir::Block;
    auto* notFunctionBlock = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(isFunction, isFunctionBlock, notFunctionBlock);

    implementBlock(builder, isFunctionBlock);
    mlir::Value result = builder.createFunctionCall(self, {self, args, kws});
    builder.create<mlir::func::ReturnOp>(result);

    implementBlock(builder, notFunctionBlock);
    result = buildSpecialMethodCall(builder, "__call__", builder.createTuplePrepend(self, args), kws);
    builder.create<mlir::func::ReturnOp>(result);
}

void buildTernaryOpCompilerBuiltin(pylir::PyBuilder& builder, llvm::StringRef functionName, llvm::StringRef method)
{
    auto func = builder.create<mlir::func::FuncOp>(
        functionName,
        builder.getFunctionType({builder.getDynamicType(), builder.getDynamicType(), builder.getDynamicType()},
                                builder.getDynamicType()));
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(func.addEntryBlock());
    auto tuple = builder.createMakeTuple({func.getArgument(0), func.getArgument(1), func.getArgument(2)}, nullptr);
    auto dict = builder.createMakeDict();
    auto result = buildSpecialMethodCall(builder, method, tuple, dict);
    builder.create<mlir::func::ReturnOp>(result);
}

void buildIOpCompilerBuiltins(pylir::PyBuilder& builder, llvm::StringRef functionName, llvm::StringRef method,
                              mlir::Value (pylir::PyBuilder::*normalOp)(mlir::Value, mlir::Value, mlir::Block*))
{
    auto func = builder.create<mlir::func::FuncOp>(
        functionName,
        builder.getFunctionType({builder.getDynamicType(), builder.getDynamicType()}, builder.getDynamicType()));
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(func.addEntryBlock());
    mlir::Value lhs = func.getArgument(0);
    mlir::Value rhs = func.getArgument(1);

    auto lhsType = builder.createTypeOf(lhs);
    auto mro = builder.createTypeMRO(lhsType);
    auto lookup = builder.createMROLookup(mro, method);
    auto failure = builder.createIsUnboundValue(lookup);
    auto* fallback = new mlir::Block;
    auto* callIOp = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(failure, fallback, callIOp);

    implementBlock(builder, callIOp);
    auto res = builder.createPylirCallIntrinsic(lookup.getResult(), builder.createMakeTuple({lhs, rhs}, nullptr),
                                                builder.createMakeDict());
    auto isNotImplemented = builder.createIs(res, builder.createNotImplementedRef());
    auto* returnBlock = new mlir::Block;
    returnBlock->addArgument(builder.getDynamicType(), builder.getCurrentLoc());
    builder.create<mlir::cf::CondBranchOp>(isNotImplemented, fallback, returnBlock, res);

    implementBlock(builder, fallback);
    res = (builder.*normalOp)(lhs, rhs, nullptr);
    isNotImplemented = builder.createIs(res, builder.createNotImplementedRef());
    auto* throwBlock = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(isNotImplemented, throwBlock, returnBlock, res);

    implementBlock(builder, throwBlock);
    auto typeError =
        pylir::buildException(builder.getCurrentLoc(), builder, pylir::Builtins::TypeError.name, {}, nullptr);
    builder.createRaise(typeError);

    implementBlock(builder, returnBlock);
    builder.create<mlir::func::ReturnOp>(builder.getCurrentLoc(), returnBlock->getArgument(0));
}

void buildGetAttributeOpCompilerBuiltin(pylir::PyBuilder& builder, llvm::StringRef functionName, llvm::StringRef method)
{
    auto func = builder.create<mlir::func::FuncOp>(
        functionName,
        builder.getFunctionType({builder.getDynamicType(), builder.getDynamicType()}, builder.getDynamicType()));
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(func.addEntryBlock());
    mlir::Value lhs = func.getArgument(0);
    mlir::Value rhs = func.getArgument(1);

    auto tuple = builder.createMakeTuple({lhs, rhs}, nullptr);
    auto dict = builder.createMakeDict();
    auto* attrError = new mlir::Block;
    attrError->addArgument(builder.getDynamicType(), builder.getCurrentLoc());
    auto result = buildSpecialMethodCall(builder, method, tuple, dict, attrError);
    builder.create<mlir::func::ReturnOp>(result);

    // If __getattribute__ raises an AttributeError we have to automatically call __getattr__.
    implementBlock(builder, attrError);
    auto exception = attrError->getArgument(0);
    auto ref = builder.createAttributeErrorRef();
    auto exceptionType = builder.createTypeOf(exception);
    auto isAttributeError = builder.createIs(exceptionType, ref);
    auto* reraiseBlock = new mlir::Block;
    auto* getattrBlock = new mlir::Block;
    builder.create<mlir::cf::CondBranchOp>(isAttributeError, getattrBlock, reraiseBlock);

    implementBlock(builder, reraiseBlock);
    builder.createRaise(exception);

    implementBlock(builder, getattrBlock);
    result = builder.createPylirGetAttrIntrinsic(lhs, rhs);
    builder.create<mlir::func::ReturnOp>(result);
}

} // namespace

void pylir::CodeGen::createCompilerBuiltinsImpl()
{
    m_builder.createGlobalValue(Builtins::None.name, true, m_builder.getObjectAttr(m_builder.getNoneTypeBuiltin()),
                                true);
    m_builder.createGlobalValue(Builtins::NotImplemented.name, true,
                                m_builder.getObjectAttr(m_builder.getNotImplementedTypeBuiltin()), true);

#define COMPILER_BUILTIN_REV_BIN_OP(name, slotName, revSlotName) \
    buildRevBinOpCompilerBuiltin(m_builder, COMPILER_BUILTIN_SLOT_TO_API_NAME(slotName), #slotName, #revSlotName);
#define COMPILER_BUILTIN_BIN_OP(name, slotName)            \
    if (#slotName != std::string_view{"__getattribute__"}) \
        buildBinOpCompilerBuiltin(m_builder, COMPILER_BUILTIN_SLOT_TO_API_NAME(slotName), #slotName);

#define COMPILER_BUILTIN_UNARY_OP(name, slotName) \
    buildUnaryOpCompilerBuiltin(m_builder, COMPILER_BUILTIN_SLOT_TO_API_NAME(slotName), #slotName);

#define COMPILER_BUILTIN_TERNARY_OP(name, slotName) \
    if (#slotName != std::string_view{"__call__"})  \
        buildTernaryOpCompilerBuiltin(m_builder, COMPILER_BUILTIN_SLOT_TO_API_NAME(slotName), #slotName);

    buildCallOpCompilerBuiltin(m_builder, "pylir__call__", "__call__");

#define COMPILER_BUILTIN_IOP(name, slotName, normalOp)                                          \
    buildIOpCompilerBuiltins(m_builder, COMPILER_BUILTIN_SLOT_TO_API_NAME(slotName), #slotName, \
                             &::pylir::PyBuilder::createPylir##normalOp##Intrinsic);

#include <pylir/Interfaces/CompilerBuiltins.def>

    buildGetAttributeOpCompilerBuiltin(m_builder, "pylir__getattribute__", "__getattribute__");
}
