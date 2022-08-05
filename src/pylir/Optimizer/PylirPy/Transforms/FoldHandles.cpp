// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/Dominance.h>
#include <mlir/IR/Matchers.h>

#include <llvm/ADT/STLExtras.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/Util/PyBuilder.hpp>

#include "PassDetail.hpp"
#include "Passes.hpp"

namespace
{
struct FoldHandlesPass : public FoldHandlesBase<FoldHandlesPass>
{
    void runOnOperation() override;
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
        auto users = userMap.getUsers(handle);
        for (auto* op : users)
        {
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
        if (!hasSingleStore)
        {
            continue;
        }

        mlir::Attribute attr;
        if (!mlir::matchPattern(singleStore.getValue(), mlir::m_Constant(&attr)))
        {
            continue;
        }

        // If the single store into the handle is already a reference to a global value there isn't a lot to be done
        // except replace all loads with such a reference. Otherwise, we create a global value with the constant as
        // initializer instead of the handle.
        auto constantStorage = attr.dyn_cast<mlir::FlatSymbolRefAttr>();
        if (!constantStorage)
        {
            constantStorage = mlir::FlatSymbolRefAttr::get(handle);
        }

        for (auto* op : llvm::make_early_inc_range(users))
        {
            // Turn any loads into constants referring to a py.globalValue
            if (!mlir::isa<pylir::Py::LoadOp>(op))
            {
                continue;
            }
            pylir::Py::PyBuilder builder(op);
            auto newOp = builder.createConstant(constantStorage);
            op->replaceAllUsesWith(newOp);
            op->erase();
        }
        singleStore->erase();

        // Create the global value if the constant was not a reference but a constant object.
        if (auto initializer = attr.dyn_cast<pylir::Py::ObjectAttrInterface>())
        {
            pylir::Py::PyBuilder builder(handle);
            auto globalValue = builder.createGlobalValue(handle.getSymName(), true, initializer);
            globalValue.setVisibility(handle.getVisibility());
        }
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

std::unique_ptr<mlir::Pass> pylir::Py::createFoldHandlesPass()
{
    return std::make_unique<FoldHandlesPass>();
}
