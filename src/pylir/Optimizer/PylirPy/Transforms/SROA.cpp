// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/Transforms/Util/SSAUpdater.hpp>

#include "PassDetail.hpp"

namespace
{
class SROA : public SROABase<SROA>
{
    llvm::DenseSet<mlir::Value> collectAggregates();

    void doAggregateReplacement(const llvm::DenseSet<mlir::Value>& aggregates);

protected:
    void runOnOperation() override;
};

llvm::DenseSet<mlir::Value> SROA::collectAggregates()
{
    llvm::DenseSet<mlir::Value> aggregates;
    getOperation()->walk(
        [&](mlir::Operation* operation)
        {
            bool valid =
                llvm::TypeSwitch<mlir::Operation*, bool>(operation)
                    .Case([](pylir::Py::MakeListOp op) { return op.getIterExpansion().empty(); })
                    .Case(
                        [](pylir::Py::MakeDictOp op)
                        {
                            if (!op.getMappingExpansion().empty())
                            {
                                return false;
                            }
                            return llvm::all_of(op.getKeys(), [](mlir::Value key)
                                                { return mlir::matchPattern(key, mlir::m_Constant()); });
                        })
                    .Case(
                        [](mlir::MemoryEffectOpInterface op)
                        {
                            // Op allocating an object that can have slots
                            llvm::SmallVector<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>> effects;
                            op.getEffects(effects);
                            if (op->getNumResults() != 1)
                            {
                                return false;
                            }
                            return llvm::all_of(effects,
                                                [&](const auto& effect) {
                                                    return effect.getValue() == op->getResult(0)
                                                           && effect.getResource()
                                                                  == mlir::SideEffects::DefaultResource::get();
                                                })
                                   || llvm::any_of(
                                       effects, [](const auto& effect)
                                       { return mlir::isa<mlir::MemoryEffects::Allocate>(effect.getEffect()); });
                        })
                    .Default(false);
            if (!valid)
            {
                return;
            }
            for (auto& iter : operation->getUses())
            {
                bool replaceAble =
                    llvm::TypeSwitch<mlir::Operation*, bool>(iter.getOwner())
                        .Case<pylir::Py::ListSetItemOp, pylir::Py::ListGetItemOp>(
                            [&](auto op)
                            {
                                if (!mlir::isa<pylir::Py::MakeListOp>(operation))
                                {
                                    return false;
                                }
                                if (iter.getOperandNumber()
                                    != static_cast<mlir::OperandRange>(op.getListMutable()).getBeginOperandIndex())
                                {
                                    return false;
                                }
                                return mlir::matchPattern(op.getIndex(), mlir::m_Constant());
                            })
                        .Case<pylir::Py::DictTryGetItemOp, pylir::Py::DictSetItemOp, pylir::Py::DictDelItemOp>(
                            [&](auto op)
                            {
                                if (!mlir::isa<pylir::Py::MakeDictOp>(operation))
                                {
                                    return false;
                                }
                                if (iter.getOperandNumber()
                                    != static_cast<mlir::OperandRange>(op.getDictMutable()).getBeginOperandIndex())
                                {
                                    return false;
                                }
                                // TODO: We currently don't allow symbol ref attrs as we have to do a lookup to
                                //       figure out their actual value and I don't yet know how I want to go about
                                //       that (especially since this is not a module pass)
                                mlir::Attribute attr;
                                if (!mlir::matchPattern(op.getKey(), mlir::m_Constant(&attr)))
                                {
                                    return false;
                                }
                                return !attr.isa<mlir::SymbolRefAttr>();
                            })
                        .Case<pylir::Py::ListLenOp, pylir::Py::ListResizeOp>(
                            [&](auto) { return mlir::isa<pylir::Py::MakeListOp>(operation); })
                        .Case<pylir::Py::GetSlotOp, pylir::Py::SetSlotOp>(
                            [&](auto op) {
                                return iter.getOperandNumber()
                                       == static_cast<mlir::OperandRange>(op.getObjectMutable()).getBeginOperandIndex();
                            })
                        // TODO: py.dict.len needs special care to implement as every py.dict.setItem we evaluate has
                        //       to also consider if such a key has previously been seen, before redefining the length
                        //       of the dictionary as incremented by one. We don't allow it until this has been
                        //       properly implemented and thought about.
                        // .Case<pylir::Py::DictLenOp>([&](auto) { return mlir::isa<pylir::Py::MakeDictOp>(operation);
                        // })
                        .Default(false);
                if (!replaceAble)
                {
                    return;
                }
            }
            aggregates.insert(operation->getResult(0));
        });
    return aggregates;
}

void SROA::doAggregateReplacement(const llvm::DenseSet<mlir::Value>& aggregates)
{
    for (auto& region : getOperation()->getRegions())
    {
        pylir::SSABuilder ssaBuilder(
            [](mlir::BlockArgument arg)
            {
                return mlir::OpBuilder::atBlockBegin(arg.getOwner())
                    .create<pylir::Py::ConstantOp>(arg.getLoc(), pylir::Py::UnboundAttr::get(arg.getContext()));
            });
        // The key is a pair consisting of the aggregate in the first position and the key/index/slot name denoting the
        // exact memory position in the second position. That way all slots in a dictionary, all indices in a list
        // and all slots in an object can be uniquely identified.
        //
        // Additionally, to keep track of the length of aggregates, the second item in the pair may be null,
        // which is a sentinel used to indicate the length of the aggregate.
        llvm::DenseMap<std::pair<mlir::Value, mlir::Attribute>, pylir::SSABuilder::DefinitionsMap> definitions;
        pylir::updateSSAinRegion(
            ssaBuilder, region,
            [&](mlir::Block* block)
            {
                for (auto& iter : llvm::make_early_inc_range(block->getOperations()))
                {
                    llvm::TypeSwitch<mlir::Operation*>(&iter)
                        .Case(
                            [&](pylir::Py::MakeListOp makeListOp)
                            {
                                if (!aggregates.contains(makeListOp))
                                {
                                    return;
                                }
                                auto indexType = mlir::IndexType::get(makeListOp.getContext());
                                for (const auto& iter : llvm::enumerate(makeListOp.getArguments()))
                                {
                                    definitions[{makeListOp, mlir::IntegerAttr::get(indexType, iter.index())}][block] =
                                        iter.value();
                                }
                                mlir::OpBuilder builder(makeListOp);
                                auto len = builder.create<mlir::arith::ConstantIndexOp>(
                                    makeListOp.getLoc(), makeListOp.getArguments().size());
                                definitions[{makeListOp, nullptr}][block] = len;
                            })
                        .Case(
                            [&](pylir::Py::ListSetItemOp setItemOp)
                            {
                                if (!aggregates.contains(setItemOp.getList()))
                                {
                                    return;
                                }
                                mlir::Attribute attr;
                                bool result = mlir::matchPattern(setItemOp.getIndex(), mlir::m_Constant(&attr));
                                PYLIR_ASSERT(result);
                                definitions[{setItemOp.getList(), attr}][block] = setItemOp.getElement();
                                setItemOp.erase();
                                m_writeOpsRemoved++;
                            })
                        .Case(
                            [&](pylir::Py::ListGetItemOp getItemOp)
                            {
                                if (!aggregates.contains(getItemOp.getList()))
                                {
                                    return;
                                }
                                mlir::Attribute attr;
                                bool result = mlir::matchPattern(getItemOp.getIndex(), mlir::m_Constant(&attr));
                                PYLIR_ASSERT(result);
                                auto replacement =
                                    ssaBuilder.readVariable(getItemOp->getLoc(), getItemOp.getType(),
                                                            definitions[{getItemOp.getList(), attr}], block);
                                getItemOp.replaceAllUsesWith(replacement);
                                getItemOp.erase();
                                m_readOpsRemoved++;
                            })
                        .Case(
                            [&](pylir::Py::ListResizeOp resizeOp)
                            {
                                if (!aggregates.contains(resizeOp.getList()))
                                {
                                    return;
                                }
                                definitions[{resizeOp.getList(), nullptr}][block] = resizeOp.getLength();
                                resizeOp.erase();
                                m_writeOpsRemoved++;
                            })
                        .Case(
                            [&](pylir::Py::ListLenOp lenOp)
                            {
                                if (!aggregates.contains(lenOp.getList()))
                                {
                                    return;
                                }
                                auto replacement = ssaBuilder.readVariable(
                                    lenOp->getLoc(), lenOp.getType(), definitions[{lenOp.getList(), nullptr}], block);
                                lenOp.replaceAllUsesWith(replacement);
                                lenOp.erase();
                                m_readOpsRemoved++;
                            })
                        .Case(
                            [&](pylir::Py::MakeDictOp makeDictOp)
                            {
                                if (!aggregates.contains(makeDictOp))
                                {
                                    return;
                                }
                                for (auto [key, value] : llvm::zip(makeDictOp.getKeys(), makeDictOp.getValues()))
                                {
                                    mlir::Attribute attr;
                                    bool result = mlir::matchPattern(key, mlir::m_Constant(&attr));
                                    PYLIR_ASSERT(result);
                                    definitions[{makeDictOp, attr}][block] = value;
                                }
                            })
                        .Case(
                            [&](pylir::Py::DictTryGetItemOp tryGetItemOp)
                            {
                                if (!aggregates.contains(tryGetItemOp.getDict()))
                                {
                                    return;
                                }
                                mlir::Attribute attr;
                                bool result = mlir::matchPattern(tryGetItemOp.getKey(), mlir::m_Constant(&attr));
                                PYLIR_ASSERT(result);
                                auto replacement =
                                    ssaBuilder.readVariable(tryGetItemOp->getLoc(), tryGetItemOp.getType(),
                                                            definitions[{tryGetItemOp.getDict(), attr}], block);
                                tryGetItemOp.replaceAllUsesWith(replacement);
                                tryGetItemOp.erase();
                                m_readOpsRemoved++;
                            })
                        .Case(
                            [&](pylir::Py::DictSetItemOp setItemOp)
                            {
                                if (!aggregates.contains(setItemOp.getDict()))
                                {
                                    return;
                                }
                                mlir::Attribute attr;
                                bool result = mlir::matchPattern(setItemOp.getKey(), mlir::m_Constant(&attr));
                                PYLIR_ASSERT(result);
                                definitions[{setItemOp.getDict(), attr}][block] = setItemOp.getValue();
                                setItemOp.erase();
                                m_writeOpsRemoved++;
                            })
                        .Case(
                            [&](pylir::Py::DictDelItemOp delItemOp)
                            {
                                if (!aggregates.contains(delItemOp.getDict()))
                                {
                                    return;
                                }
                                mlir::Attribute attr;
                                bool result = mlir::matchPattern(delItemOp.getKey(), mlir::m_Constant(&attr));
                                PYLIR_ASSERT(result);
                                mlir::OpBuilder builder(delItemOp);
                                auto unbound = builder.create<pylir::Py::ConstantOp>(
                                    delItemOp->getLoc(), pylir::Py::UnboundAttr::get(delItemOp->getContext()));
                                definitions[{delItemOp.getDict(), attr}][block] = unbound;
                                delItemOp.erase();
                                m_writeOpsRemoved++;
                            })
                        .Case(
                            [&](pylir::Py::SetSlotOp setSlotOp)
                            {
                                if (!aggregates.contains(setSlotOp.getObject()))
                                {
                                    return;
                                }
                                definitions[{setSlotOp.getObject(), setSlotOp.getSlotAttr()}][block] =
                                    setSlotOp.getValue();
                                setSlotOp.erase();
                                m_writeOpsRemoved++;
                            })
                        .Case(
                            [&](pylir::Py::GetSlotOp getSlotOp)
                            {
                                if (!aggregates.contains(getSlotOp.getObject()))
                                {
                                    return;
                                }
                                auto replacement = ssaBuilder.readVariable(
                                    getSlotOp->getLoc(), getSlotOp.getType(),
                                    definitions[{getSlotOp.getObject(), getSlotOp.getSlotAttr()}], block);
                                getSlotOp.replaceAllUsesWith(replacement);
                                getSlotOp.erase();
                                m_readOpsRemoved++;
                            });
                }
            });
    }
    for (auto iter : aggregates)
    {
        iter.getDefiningOp()->erase();
        m_aggregatesRemoved++;
    }
}

void SROA::runOnOperation()
{
    bool changed = false;
    bool changedThisIteration;
    do
    {
        changedThisIteration = false;
        auto aggregates = collectAggregates();
        if (!aggregates.empty())
        {
            changedThisIteration = true;
        }
        else
        {
            continue;
        }
        doAggregateReplacement(aggregates);
    } while (changedThisIteration);
    if (!changed)
    {
        markAllAnalysesPreserved();
    }
}

} // namespace

std::unique_ptr<mlir::Pass> pylir::Py::createSROAPass()
{
    return std::make_unique<SROA>();
}
