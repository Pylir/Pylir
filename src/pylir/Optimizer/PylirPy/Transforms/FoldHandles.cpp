// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/Dominance.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/Transforms/Util/SSAUpdater.hpp>

#include "Passes.hpp"

namespace pylir::Py
{
#define GEN_PASS_DEF_FOLDHANDLESPASS
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"
} // namespace pylir::Py

namespace
{
struct FoldHandlesPass : public pylir::Py::impl::FoldHandlesPassBase<FoldHandlesPass>
{
    using Base::Base;

protected:
    void runOnOperation() override;

private:
    pylir::Py::GlobalValueOp createGlobalValueFromHandle(pylir::Py::GlobalHandleOp handleOp,
                                                         pylir::Py::ObjectAttrInterface initializer, bool constant)
    {
        mlir::OpBuilder builder(handleOp);
        return builder.create<pylir::Py::GlobalValueOp>(handleOp->getLoc(), handleOp.getSymName(),
                                                        handleOp.getSymVisibilityAttr(), constant, initializer);
    }

    void replaceLoadsWithAttr(llvm::ArrayRef<mlir::Operation*> users, mlir::Attribute constant)
    {
        for (auto* op : llvm::make_early_inc_range(users))
        {
            // Turn any loads into constants referring to a py.globalValue
            if (!mlir::isa<pylir::Py::LoadOp>(op))
            {
                continue;
            }
            mlir::OpBuilder builder(op);
            auto newOp = builder.create<pylir::Py::ConstantOp>(op->getLoc(), constant);
            op->replaceAllUsesWith(newOp);
            op->erase();
        }
    }

    void handleSingleStoreConstant(mlir::Attribute attr, pylir::Py::StoreOp singleStore,
                                   pylir::Py::GlobalHandleOp handle, llvm::ArrayRef<mlir::Operation*> users)
    {
        // If the single store into the handle is already a reference to a global value there isn't a lot to be done
        // except replace all loads with such a reference. Otherwise if not unbound, we create a global value with the
        // constant as initializer instead of the handle.
        mlir::Attribute constantStorage = attr.dyn_cast<mlir::FlatSymbolRefAttr>();
        if (!constantStorage)
        {
            constantStorage = attr.dyn_cast<pylir::Py::UnboundAttr>();
        }
        if (!constantStorage)
        {
            constantStorage = mlir::FlatSymbolRefAttr::get(handle);
        }

        replaceLoadsWithAttr(users, constantStorage);
        singleStore->erase();

        // Create the global value if the constant was not a reference but a constant object.
        if (auto initializer = attr.dyn_cast<pylir::Py::ObjectAttrInterface>())
        {
            createGlobalValueFromHandle(handle, initializer, true);
        }
        handle->erase();
        m_singleStoreHandlesConverted++;
    }

    void handleSingleFunctionHandle(mlir::Region& parent, pylir::Py::GlobalHandleOp handleOp)
    {
        pylir::SSABuilder builder(
            [](mlir::Block* block, mlir::Location loc) -> mlir::Value
            {
                auto builder = mlir::OpBuilder::atBlockBegin(block);
                return builder.create<pylir::Py::ConstantOp>(loc, builder.getAttr<pylir::Py::UnboundAttr>());
            });
        pylir::SSABuilder::DefinitionsMap definitions;
        pylir::updateSSAinRegion(builder, parent,
                                 [&](mlir::Block* block)
                                 {
                                     for (auto& op : llvm::make_early_inc_range(*block))
                                     {
                                         if (auto store = mlir::dyn_cast<pylir::Py::StoreOp>(op))
                                         {
                                             if (store.getHandleAttr().getAttr() != handleOp.getSymNameAttr())
                                             {
                                                 continue;
                                             }
                                             definitions[block] = store.getValue();
                                             store->erase();
                                             continue;
                                         }
                                         if (auto load = mlir::dyn_cast<pylir::Py::LoadOp>(op))
                                         {
                                             if (load.getHandleAttr().getAttr() != handleOp.getSymNameAttr())
                                             {
                                                 continue;
                                             }
                                             load.replaceAllUsesWith(builder.readVariable(
                                                 load->getLoc(), load.getType(), definitions, block));
                                             load->erase();
                                             continue;
                                         }
                                     }
                                 });
    }
};

void FoldHandlesPass::runOnOperation()
{
    auto module = getOperation();
    mlir::SymbolTableCollection collection;
    mlir::SymbolUserMap userMap(collection, module);
    bool changed = false;
    for (auto handle : llvm::make_early_inc_range(module.getOps<pylir::Py::GlobalHandleOp>()))
    {
        // If the globalHandle is not public and there is only a single store to it with a constant value,
        // change it to a globalValueOp. If there are no loads, remove it entirely.
        if (handle.isPublic())
        {
            continue;
        }
        pylir::Py::StoreOp singleStore;
        bool hasSingleStore = false;
        bool hasLoads = false;
        bool hasSingleParent = false;
        mlir::Region* singleParent = nullptr;

        auto users = userMap.getUsers(handle);
        for (auto* op : users)
        {
            if (!singleParent)
            {
                singleParent = op->getParentRegion();
                hasSingleParent = true;
            }
            else if (singleParent != op->getParentRegion())
            {
                hasSingleParent = false;
            }

            if (auto storeOp = mlir::dyn_cast<pylir::Py::StoreOp>(op))
            {
                if (!singleStore)
                {
                    hasSingleStore = true;
                    singleStore = storeOp;
                }
                else
                {
                    hasSingleStore = false;
                }
            }
            if (mlir::isa<pylir::Py::LoadOp>(op))
            {
                hasLoads = true;
            }
        }
        // Remove if it has no loads
        if (!hasLoads)
        {
            m_noLoadHandlesRemoved++;
            std::for_each(users.begin(), users.end(), std::mem_fn(&mlir::Operation::erase));
            handle->erase();
            changed = true;
            continue;
        }
        if (hasSingleParent)
        {
            m_singleRegionHandlesConverted++;
            handleSingleFunctionHandle(*singleParent, handle);
            handle->erase();
            changed = true;
            continue;
        }
        if (!hasSingleStore)
        {
            continue;
        }

        mlir::Attribute attr;
        auto value = singleStore.getValue();
        if (mlir::matchPattern(value, mlir::m_Constant(&attr)))
        {
            handleSingleStoreConstant(attr, singleStore, handle, users);
            changed = true;
            continue;
        }
        auto* op = value.getDefiningOp();
        if (!op)
        {
            continue;
        }
        auto ref =
            llvm::TypeSwitch<mlir::Operation*, mlir::FlatSymbolRefAttr>(op)
                .Case(
                    [&](pylir::Py::MakeFuncOp makeFuncOp)
                    {
                        auto value = createGlobalValueFromHandle(
                            handle, pylir::Py::FunctionAttr::get(&getContext(), makeFuncOp.getFunctionAttr()), false);
                        mlir::OpBuilder builder(makeFuncOp);
                        auto ref = mlir::FlatSymbolRefAttr::get(value);
                        auto c = builder.create<pylir::Py::ConstantOp>(makeFuncOp->getLoc(), ref);
                        makeFuncOp->replaceAllUsesWith(c);
                        makeFuncOp->erase();
                        return ref;
                    })
                .Default({nullptr});
        if (!ref)
        {
            continue;
        }
        replaceLoadsWithAttr(users, ref);
        singleStore->erase();
        handle->erase();
        m_singleStoreHandlesConverted++;
        changed = true;
    }
    if (!changed)
    {
        markAllAnalysesPreserved();
    }
    markAnalysesPreserved<mlir::DominanceInfo>();
}

} // namespace
