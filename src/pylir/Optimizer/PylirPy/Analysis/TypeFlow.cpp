// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypeFlow.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Analysis/LoopInfo.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/TypeRefineableInterface.hpp>

namespace
{

class ValueTracking
{
    llvm::DenseMap<mlir::Value, mlir::Value> m_defs;
    pylir::TypeFlow::UndefOp& m_undefOp;

public:
    explicit ValueTracking(pylir::TypeFlow::UndefOp& undefOp) : m_undefOp(undefOp) {}

    void def(mlir::Value old, mlir::Value newValue)
    {
        m_defs[old] = newValue;
    }

    [[nodiscard]] mlir::Value use(mlir::Value old)
    {
        auto val = m_defs.lookup(old);
        return val ? val : m_undefOp;
    }

    [[nodiscard]] pylir::TypeFlow::UndefOp getUndef() const
    {
        return m_undefOp;
    }
};

void handleBranchOpInterface(mlir::BranchOpInterface branchOpInterface, mlir::ImplicitLocOpBuilder& builder,
                             ValueTracking& valueTracking, llvm::function_ref<mlir::Block*(mlir::Block*)> blockMapping,
                             const llvm::DenseSet<mlir::Value>& foldEdges)
{
    llvm::SmallVector<llvm::SmallVector<mlir::Value>> args(branchOpInterface->getNumSuccessors());
    for (std::size_t i = 0; i < args.size(); i++)
    {
        auto ops = branchOpInterface.getSuccessorOperands(i);
        for (std::size_t j = 0; j < ops.size(); j++)
        {
            args[i].push_back(valueTracking.use(ops[j]));
        }
    }

    if (auto cond = mlir::dyn_cast<mlir::cf::CondBranchOp>(*branchOpInterface))
    {
        if (foldEdges.contains(cond.getCondition()))
        {
            builder.create<pylir::TypeFlow::CondBranchOp>(valueTracking.use(cond.getCondition()), args[0], args[1],
                                                          blockMapping(cond.getTrueDest()),
                                                          blockMapping(cond.getFalseDest()));
            return;
        }
    }

    llvm::SmallVector<mlir::ValueRange> ranges(args.begin(), args.end());
    builder.create<pylir::TypeFlow::BranchOp>(
        ranges, llvm::to_vector(llvm::map_range(branchOpInterface->getSuccessors(), blockMapping)));
}

void handleNoReadWriteEffectOp(mlir::Operation* op, mlir::ImplicitLocOpBuilder& builder, ValueTracking& valueTracking,
                          llvm::DenseSet<mlir::Value>& foldEdges)
{
    auto typeDependentConstant =
        llvm::any_of(op->getOperands(), [&](mlir::Value val) { return foldEdges.contains(val); });
    if (!mlir::isa<pylir::Py::TypeRefineableInterface>(op) && !typeDependentConstant)
    {
        return;
    }
    auto res = builder.create<pylir::TypeFlow::CalcOp>(
        llvm::to_vector(
            llvm::map_range(op->getOperands(), [&](mlir::Value value) { return valueTracking.use(value); })),
        op, typeDependentConstant);
    for (auto [prev, iter] : llvm::zip(op->getResults(), res.getResults()))
    {
        // If the reason this op is being added is that it might be folded by value depended
        // on a type of, we add its results regardless of its type.
        if (!typeDependentConstant && !prev.getType().isa<pylir::Py::DynamicType>())
        {
            continue;
        }
        valueTracking.def(prev, iter);
        if (typeDependentConstant)
        {
            foldEdges.insert(prev);
        }
    }
}

void dispatchOperations(mlir::Operation* op, mlir::ImplicitLocOpBuilder& builder, ValueTracking& valueTracking,
                        llvm::function_ref<mlir::Block*(mlir::Block*)> blockMapping,
                        llvm::DenseSet<mlir::Value>& foldEdges)
{
    mlir::Attribute attr;
    if (mlir::matchPattern(op, mlir::m_Constant(&attr)))
    {
        auto constant = builder.create<pylir::TypeFlow::ConstantOp>(attr);
        valueTracking.def(op->getResult(0), constant);
        return;
    }

    auto mapOperands = [&](mlir::ValueRange values)
    {
        llvm::SmallVector<mlir::Value> result;
        for (auto iter : values)
        {
            if (!iter.getType().isa<pylir::Py::DynamicType>())
            {
                continue;
            }
            result.push_back(valueTracking.use(iter));
        }
        return result;
    };

    llvm::TypeSwitch<mlir::Operation*>(op)
        .Case(
            [&](pylir::Py::TypeOfOp typeOf)
            {
                auto newTypeOf = builder.create<pylir::TypeFlow::TypeOfOp>(valueTracking.use(typeOf.getObject()));
                valueTracking.def(typeOf, newTypeOf);
                foldEdges.insert(typeOf);
            })
        .Case(
            [&](pylir::Py::ObjectFromTypeObjectInterface fromTypeObjectInterface)
            {
                auto newMakeObject = builder.create<pylir::TypeFlow::MakeObjectOp>(
                    valueTracking.use(fromTypeObjectInterface.getTypeObject()));
                valueTracking.def(fromTypeObjectInterface->getResult(0), newMakeObject);
            })
        .Case(
            [&](mlir::CallOpInterface callOp)
            {
                auto callable = callOp.getCallableForCallee();
                PYLIR_ASSERT(callable);
                auto mapOutputs = [&](auto newCall)
                {
                    std::size_t i = 0;
                    for (auto iter : callOp->getResults())
                    {
                        if (!iter.getType().isa<pylir::Py::DynamicType>())
                        {
                            continue;
                        }
                        valueTracking.def(iter, newCall.getResults()[i++]);
                    }
                };
                if (auto ref = callable.dyn_cast<mlir::SymbolRefAttr>())
                {
                    auto newCall = builder.create<pylir::TypeFlow::CallOp>(
                        std::vector<mlir::Type>(llvm::count_if(callOp->getResultTypes(),
                                                               std::mem_fn(&mlir::Type::isa<pylir::Py::DynamicType>)),
                                                builder.getType<pylir::TypeFlow::MetaType>()),
                        ref, mapOperands(callOp.getArgOperands()));
                    mapOutputs(newCall);
                }
                else
                {
                    auto newCall = builder.create<pylir::TypeFlow::CallIndirectOp>(
                        std::vector<mlir::Type>(llvm::count_if(callOp->getResultTypes(),
                                                               std::mem_fn(&mlir::Type::isa<pylir::Py::DynamicType>)),
                                                builder.getType<pylir::TypeFlow::MetaType>()),
                        valueTracking.use(callable.get<mlir::Value>()), mapOperands(callOp.getArgOperands()));
                    mapOutputs(newCall);
                }
                auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>(*callOp);
                if (!branchOp)
                {
                    return;
                }
                handleBranchOpInterface(branchOp, builder, valueTracking, blockMapping, foldEdges);
            })
        .Case([&](mlir::BranchOpInterface branchOpInterface)
              { handleBranchOpInterface(branchOpInterface, builder, valueTracking, blockMapping, foldEdges); })
        .Default(
            [&](mlir::Operation* op)
            {
                if (op->hasTrait<mlir::OpTrait::ReturnLike>())
                {
                    builder.create<pylir::TypeFlow::ReturnOp>(mapOperands(op->getOperands()));
                    return;
                }
                if (op->hasTrait<mlir::OpTrait::IsTerminator>())
                {
                    llvm::SmallVector<llvm::SmallVector<mlir::Value>> args(op->getNumSuccessors());
                    for (std::size_t i = 0; i < args.size(); i++)
                    {
                        args[i].resize(op->getSuccessor(i)->getNumArguments(), valueTracking.getUndef());
                    }
                    llvm::SmallVector<mlir::ValueRange> ranges(args.begin(), args.end());
                    builder.create<pylir::TypeFlow::BranchOp>(
                        ranges, llvm::to_vector(llvm::map_range(op->getSuccessors(), blockMapping)));
                    return;
                }

                llvm::SmallVector<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> effects;
                if (auto effectInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op))
                {
                    effectInterface.getEffects(effects);
                }
                // TODO: Handle more than one read and write on an operand if necessary
                auto* result =
                    llvm::find_if(effects,
                                  [op](const mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>& effect)
                                  {
                                      auto val = effect.getValue();
                                      return val && val.getDefiningOp() != op
                                             && llvm::is_contained<mlir::SideEffects::Effect*>(
                                                 {mlir::MemoryEffects::Read::get(), mlir::MemoryEffects::Write::get()},
                                                 effect.getEffect());
                                  });
                if (result == effects.end())
                {
                    handleNoReadWriteEffectOp(op, builder, valueTracking, foldEdges);
                    return;
                }
            });
}

#define GEN_PASS_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowPasses.h.inc"

class TypeFlowLoopLift : public TypeFlowLoopLiftBase<TypeFlowLoopLift>
{
protected:
    void runOnOperation() override;
};

} // namespace

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

    pylir::TypeFlow::UndefOp undef;
    ValueTracking valueTracking(undef);

    llvm::DenseMap<mlir::Block*, mlir::Block*> blockMapping;

    auto* entryBlock = m_function->addEntryBlock();
    blockMapping.insert({&func.front(), entryBlock});
    std::size_t i = 0;
    for (auto iter : func.getArguments())
    {
        if (iter.getType().isa<pylir::Py::DynamicType>())
        {
            valueTracking.def(iter, entryBlock->getArgument(i++));
        }
    }

    auto getBlockMapping = [&](mlir::Block* input)
    {
        auto [result, inserted] = blockMapping.insert({input, nullptr});
        if (inserted)
        {
            result->second = new mlir::Block;
        }
        return result->second;
    };

    llvm::DenseSet<mlir::Value> foldEdges;
    for (auto& block : func.getBody())
    {
        auto* mappedBlock = getBlockMapping(&block);
        if (!mappedBlock->getParent())
        {
            m_function->push_back(mappedBlock);
        }

        builder.setInsertionPointToStart(mappedBlock);
        if (m_function->back().isEntryBlock())
        {
            undef = builder.create<pylir::TypeFlow::UndefOp>();
        }
        else
        {
            for (auto iter : block.getArguments())
            {
                valueTracking.def(iter, mappedBlock->addArgument(type, builder.getLoc()));
            }
        }
        blockMapping.insert({&block, mappedBlock});
        for (auto& op : block)
        {
            dispatchOperations(&op, builder, valueTracking, getBlockMapping, foldEdges);
        }
    }

    mlir::PassManager passManager(m_function->getContext(), m_function->getOperationName());
    passManager.addPass(std::make_unique<TypeFlowLoopLift>());
    auto result = passManager.run(*m_function);
    PYLIR_ASSERT(mlir::succeeded(result));
}

void pylir::Py::TypeFlow::dump()
{
    m_function->dump();
}

void TypeFlowLoopLift::runOnOperation()
{
    auto& loopInfo = getAnalysis<pylir::LoopInfo>();
    for (auto* topLevelLoop : loopInfo.getTopLevelLoops())
    {
        // Perform a preorder walk via a worklist
        llvm::SmallVector<pylir::Loop*> worklist{topLevelLoop};
        while (!worklist.empty())
        {
            auto* loop = worklist.pop_back_val();
            worklist.append(loop->begin(), loop->end());
            auto* loopHeader = loop->getHeader();

            auto loopEntries = llvm::to_vector(llvm::make_filter_range(
                loop->getHeader()->getPredecessors(), [loop](mlir::Block* pred) { return !loop->contains(pred); }));
            mlir::Block* preHeader = llvm::hasSingleElement(loopEntries) ? loopEntries.front() : nullptr;
            llvm::SmallVector<mlir::Value> regionInit;
            if (!preHeader || preHeader->getNumSuccessors() != 1
                || !mlir::isa<pylir::TypeFlow::BranchOp>(preHeader->getTerminator()))
            {
                // If there is no single pre-header that dominates the header or the pre-headers terminator is not a
                // noop branch we need to create one that will be the successor of all the loop header predecessors and
                // will contain the loop region
                preHeader = new mlir::Block;
                preHeader->insertBefore(loopHeader);
                for (auto& iter : loopHeader->getArguments())
                {
                    preHeader->addArgument(iter.getType(), iter.getLoc());
                }
                for (auto& iter : loopHeader->getUses())
                {
                    if (loop->contains(iter.getOwner()->getBlock()))
                    {
                        continue;
                    }
                    iter.set(preHeader);
                }
                regionInit.append(preHeader->args_begin(), preHeader->args_end());
            }
            else
            {
                auto terminator = mlir::cast<pylir::TypeFlow::BranchOp>(preHeader->getTerminator());
                auto branchArgs = terminator.getBranchArgs()[0];
                regionInit.append(branchArgs.begin(), branchArgs.end());
                terminator->erase();
            }

            llvm::MapVector<mlir::Block*, std::uint32_t> successors;
            mlir::Region loopRegion;
            for (auto* block : loop->getBlocks())
            {
                block->getParent()->getBlocks().remove(block);
                loopRegion.push_back(block);
                auto terminator = mlir::cast<mlir::BranchOpInterface>(block->getTerminator());
                // If this is dummy back edge
                if (terminator && terminator->getSuccessors() == llvm::ArrayRef<mlir::Block*>{loopHeader})
                {
                    auto builder = mlir::OpBuilder(terminator);
                    builder.create<pylir::TypeFlow::YieldOp>(terminator->getLoc(),
                                                             terminator.getSuccessorOperands(0).getForwardedOperands());
                    terminator.erase();
                    continue;
                }

                for (auto& iter : llvm::enumerate(terminator->getSuccessors()))
                {
                    if (iter.value() != loopHeader && loop->contains(iter.value()))
                    {
                        continue;
                    }
                    auto* trampoline = new mlir::Block;
                    loopRegion.push_back(trampoline);
                    mlir::OpBuilder builder(&getContext());
                    builder.setInsertionPointToStart(trampoline);
                    auto successorOperands = terminator.getSuccessorOperands(iter.index());
                    PYLIR_ASSERT(successorOperands.getProducedOperandCount() == 0);
                    auto args = successorOperands.getForwardedOperands();
                    if (iter.value() == loopHeader)
                    {
                        builder.create<pylir::TypeFlow::YieldOp>(terminator->getLoc(), args);
                    }
                    else
                    {
                        builder.create<pylir::TypeFlow::ExitOp>(
                            terminator->getLoc(), successors.insert({iter.value(), successors.size()}).first->second,
                            args);
                    }
                    terminator->setSuccessor(trampoline, iter.index());
                    successorOperands.erase(0, args.size());
                }
            }

            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPointToEnd(preHeader);
            auto loopOp = builder.create<pylir::TypeFlow::LoopRegionOp>(
                builder.getUnknownLoc(), regionInit, llvm::to_vector(llvm::make_first_range(successors)));
            loopOp.getBody().takeBody(loopRegion);
        }
    }

    if (loopInfo.getTopLevelLoops().empty())
    {
        markAllAnalysesPreserved();
    }
}
