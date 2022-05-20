// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypeFlow.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Matchers.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/TypeRefineableInterface.hpp>

pylir::Py::TypeFlow::TypeFlow(mlir::Operation* operation)
{
    auto func = mlir::cast<mlir::FunctionOpInterface>(operation);

    mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(func.getContext()), operation->getContext());
    mlir::OperationState state(builder.getLoc(), pylir::TypeFlow::FuncOp::getOperationName());
    auto type = builder.getType<pylir::TypeFlow::MetaType>();
    auto functionType = builder.getFunctionType(
        std::vector<mlir::Type>(
            llvm::count_if(func.getArgumentTypes(), std::mem_fn(&mlir::Type::isa<pylir::Py::DynamicType>)), type),
        std::vector<mlir::Type>(
            llvm::count_if(func.getResultTypes(), std::mem_fn(&mlir::Type::isa<pylir::Py::DynamicType>)), type));
    pylir::TypeFlow::FuncOp::build(
        builder, state, func.getName(), functionType,
        builder.getArrayAttr(std::vector<mlir::Attribute>(functionType.getNumInputs(), builder.getDictionaryAttr({}))),
        builder.getArrayAttr(
            std::vector<mlir::Attribute>(functionType.getNumResults(), builder.getDictionaryAttr({}))));
    m_function = mlir::cast<pylir::TypeFlow::FuncOp>(builder.create(state));

    llvm::DenseMap<mlir::Value, mlir::Value> valueMapping;
    llvm::DenseMap<mlir::Block*, mlir::Block*> blockMapping;
    llvm::DenseSet<mlir::Value> foldEdges;
    pylir::TypeFlow::UndefOp undef;

    auto getValueMapping = [&](mlir::Value input)
    {
        auto lookup = valueMapping.lookup(input);
        return lookup ? lookup : undef;
    };

    auto getBlockMapping = [&](mlir::Block* input)
    {
        auto [result, inserted] = blockMapping.insert({input, nullptr});
        if (inserted)
        {
            result->second = new mlir::Block;
        }
        return result->second;
    };

    for (auto& block : func.getBody())
    {
        auto [result, inserted] = blockMapping.insert({&block, nullptr});
        if (inserted)
        {
            result->second = m_function->empty() ? m_function->addEntryBlock() : m_function->addBlock();
        }
        else
        {
            m_function->push_back(result->second);
        }
        auto* mappedBlock = result->second;
        builder.setInsertionPointToStart(mappedBlock);
        if (m_function->back().isEntryBlock())
        {
            undef = builder.create<pylir::TypeFlow::UndefOp>();
        }
        for (auto iter : block.getArguments())
        {
            // Entry block values may only exist for type objects. If we were to do the abstract interpretation using
            // all possible values that'd be quite insane. We only allow types (both as types of objects, and type
            // objects) as inputs. So filter out anything that is not an ObjectTypeInterface
            if (mappedBlock->isEntryBlock() && !iter.getType().isa<pylir::Py::ObjectTypeInterface>())
            {
                continue;
            }
            valueMapping.insert({iter, mappedBlock->addArgument(iter.getType(), builder.getLoc())});
        }
        blockMapping.insert({&block, mappedBlock});
        for (auto& op : block)
        {
            mlir::Attribute attr;
            if (mlir::matchPattern(&op, mlir::m_Constant(&attr)))
            {
                auto constant = builder.create<pylir::TypeFlow::ConstantOp>(attr);
                valueMapping.insert({op.getResult(0), constant});
                continue;
            }
            llvm::TypeSwitch<mlir::Operation*>(&op)
                .Case(
                    [&](pylir::Py::TypeOfOp typeOf)
                    {
                        auto lookup = valueMapping.lookup(typeOf.getObject());
                        if (!lookup)
                        {
                            return;
                        }
                        auto typeOfOp = builder.create<pylir::TypeFlow::TypeOfOp>(lookup);
                        valueMapping.insert({typeOf, typeOfOp});
                        foldEdges.insert(typeOf);
                    })
                .Case(
                    [&](mlir::BranchOpInterface branchOpInterface)
                    {
                        llvm::SmallVector<llvm::SmallVector<mlir::Value>> args(branchOpInterface->getNumSuccessors());
                        for (std::size_t i = 0; i < args.size(); i++)
                        {
                            auto ops = branchOpInterface.getSuccessorOperands(i);
                            for (std::size_t j = 0; j < ops.size(); j++)
                            {
                                args[i].push_back(getValueMapping(ops[j]));
                            }
                        }

                        if (auto cond = mlir::dyn_cast<mlir::cf::CondBranchOp>(*branchOpInterface))
                        {
                            if (foldEdges.contains(cond.getCondition()))
                            {
                                builder.create<pylir::TypeFlow::CondBranchOp>(
                                    valueMapping.lookup(cond.getCondition()), args[0], args[1],
                                    getBlockMapping(cond.getTrueDest()), getBlockMapping(cond.getFalseDest()));
                                return;
                            }
                        }

                        llvm::SmallVector<mlir::ValueRange> ranges(args.begin(), args.end());
                        auto successors = llvm::to_vector(branchOpInterface->getSuccessors());
                        llvm::for_each(successors, [&](mlir::Block*& block) { block = getBlockMapping(block); });
                        builder.create<pylir::TypeFlow::BranchOp>(ranges, successors);
                    })
                .Default(
                    [&](mlir::Operation* op)
                    {
                        if (op->hasTrait<mlir::OpTrait::ReturnLike>())
                        {
                            builder.create<pylir::TypeFlow::ReturnOp>(
                                llvm::to_vector(llvm::map_range(op->getOperands(), getValueMapping)));
                            return;
                        }
                        if (op->hasTrait<mlir::OpTrait::IsTerminator>())
                        {
                            llvm::SmallVector<llvm::SmallVector<mlir::Value>> args(op->getNumSuccessors());
                            for (std::size_t i = 0; i < args.size(); i++)
                            {
                                args[i].resize(op->getSuccessor(i)->getNumArguments(), undef);
                            }
                            llvm::SmallVector<mlir::ValueRange> ranges(args.begin(), args.end());
                            auto successors = llvm::to_vector(op->getSuccessors());
                            llvm::for_each(successors, [&](mlir::Block*& block) { block = getBlockMapping(block); });
                            builder.create<pylir::TypeFlow::BranchOp>(ranges, successors);
                            return;
                        }

                        auto isRefineable = mlir::isa<pylir::Py::TypeRefineableInterface>(op);
                        if (!isRefineable
                            && llvm::none_of(op->getOperands(),
                                             [&](mlir::Value val) { return foldEdges.contains(val); }))
                        {
                            return;
                        }
                        auto res = builder.create<pylir::TypeFlow::CalcOp>(
                            llvm::to_vector(llvm::map_range(op->getOperands(), getValueMapping)), op);
                        for (auto [prev, iter] : llvm::zip(op->getResults(), res.getResults()))
                        {
                            // If the reason this op is being added is because it might be fold by value depended
                            // on a type of, we add its results regardless of its type.
                            if (!isRefineable && !prev.getType().isa<pylir::Py::DynamicType>())
                            {
                                continue;
                            }
                            valueMapping.insert({prev, iter});
                            if (!isRefineable)
                            {
                                foldEdges.insert(prev);
                            }
                        }
                    });
        }
    }
}
