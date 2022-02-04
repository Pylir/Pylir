#include <mlir/IR/Dominance.h>
#include <mlir/IR/Matchers.h>

#include <llvm/ADT/STLExtras.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/Util/PyBuilder.hpp>

#include "PassDetail.hpp"
#include "Passes.hpp"

namespace
{
struct FoldHandlesPass : public pylir::Py::FoldHandlesBase<FoldHandlesPass>
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
        for (auto *op : users)
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
        pylir::Py::ObjectAttr attr;
        if (!mlir::matchPattern(singleStore.value(), mlir::m_Constant(&attr)))
        {
            continue;
        }
        for (auto *op : llvm::make_early_inc_range(users))
        {
            // Turn any loads into constants referring to a py.globalValue
            if (!mlir::isa<pylir::Py::LoadOp>(op))
            {
                continue;
            }
            pylir::Py::PyBuilder builder(op);
            auto newOp = builder.createConstant(mlir::FlatSymbolRefAttr::get(handle));
            op->replaceAllUsesWith(newOp);
            op->erase();
        }
        singleStore->erase();
        pylir::Py::PyBuilder builder(handle);
        auto globalValue = builder.createGlobalValue(handle.sym_name(), false, attr);
        globalValue.setVisibility(handle.getVisibility());
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
