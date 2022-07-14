// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Analysis/CallGraph.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/STLExtras.h>

#include <pylir/Optimizer/Analysis/BodySize.hpp>

#include <numeric>
#include <queue>

#include "PassDetail.hpp"
#include "Passes.hpp"
#include "Util/InlinerUtil.hpp"

namespace
{
class Inliner : public InlinerBase<Inliner>
{
    mlir::FrozenRewritePatternSet patterns;

protected:
    mlir::LogicalResult initialize(mlir::MLIRContext* context) override
    {
        mlir::RewritePatternSet set(context);
        for (auto* dialect : context->getLoadedDialects())
        {
            dialect->getCanonicalizationPatterns(set);
        }
        for (auto& op : context->getRegisteredOperations())
        {
            op.getCanonicalizationPatterns(set, context);
        }
        patterns = mlir::FrozenRewritePatternSet(std::move(set));
        return mlir::success();
    }

    void runOnOperation() override
    {
        mlir::SymbolTableCollection collection;
        struct CallSite
        {
            std::int32_t costBenefit;
            mlir::CallOpInterface callOp;
            mlir::CallableOpInterface callableOp;
            mlir::CallableOpInterface containedCallable;
            std::size_t generation;

            bool operator>(const CallSite& rhs) const noexcept
            {
                return costBenefit > rhs.costBenefit;
            }
        };
        std::priority_queue<CallSite, std::vector<CallSite>, std::greater<CallSite>> callSites;

        llvm::DenseSet<mlir::Operation*> seenCalls;
        llvm::DenseMap<mlir::Operation*, llvm::DenseSet<mlir::Operation*>> callees;

        auto addCallOps = [&](
                              mlir::Operation* top, std::size_t generation = 0,
                              llvm::function_ref<bool(mlir::CallOpInterface, mlir::CallableOpInterface)> filterAction =
                                  [](mlir::CallOpInterface, mlir::CallableOpInterface) { return true; })
        {
            top->walk(
                [&](mlir::CallOpInterface callOpInterface)
                {
                    if (!seenCalls.insert(callOpInterface).second)
                    {
                        return;
                    }
                    auto callable =
                        mlir::dyn_cast_or_null<mlir::CallableOpInterface>(callOpInterface.resolveCallable(&collection));
                    if (!callable || !callable.getCallableRegion() || !filterAction(callOpInterface, callable))
                    {
                        return;
                    }
                    callees[callable].insert(callOpInterface);
                    callSites.push({costBenefitHeuristic(callOpInterface, callable), callOpInterface, callable,
                                    callOpInterface->getParentOfType<mlir::CallableOpInterface>(), generation});
                });
        };

        addCallOps(getOperation());

        struct CallableData
        {
            std::size_t initialSize;
            std::size_t currentSize;
            std::size_t generation;
        };
        llvm::DenseMap<mlir::Operation*, CallableData> callableData;
        getOperation()->walk(
            [&](mlir::CallableOpInterface callableOpInterface)
            {
                auto size = getChildAnalysis<pylir::BodySize>(callableOpInterface).getSize();
                callableData.insert({callableOpInterface, {size, size, 0}});
            });

        llvm::DenseSet<mlir::Operation*> invalidatedCalls;
        llvm::DenseMap<mlir::Operation*, llvm::DenseMap<mlir::Operation*, std::size_t>> callableInlinedCallables;
        while (!callSites.empty())
        {
            auto callSite = callSites.top();
            callSites.pop();
            auto containedCallable = callSite.containedCallable;
            auto& info = callableData[containedCallable];
            // This callsite is out of date and likely doesn't exist anymore.
            if (info.generation != callSite.generation)
            {
                continue;
            }
            // The cost benefit of this callsite has changed due to the callers size having changed. Recompute and
            // reinsert into the priority queue.
            if (invalidatedCalls.erase(callSite.callOp))
            {
                callSite.costBenefit = costBenefitHeuristic(callSite.callOp, callSite.callableOp);
                callSites.push(callSite);
                continue;
            }

            // TODO: MLIR does not yet support inlining a region into itself, hence we can't yet handle direct
            // recursion
            if (containedCallable == callSite.callableOp)
            {
                continue;
            }

            if (info.currentSize + callSite.costBenefit > info.initialSize * m_maxFuncGrowth / 100)
            {
                continue;
            }

            // Due to folding and pattern application a callsite in the callee might get deleted
            // To handle this we delete all of them in the current function and readd them during the rescan phase that
            // we'd have to do anyways to discover new call sites after inlining.
            //
            // Note this does not remove the callsites out of the priority queue. Ideally we'd be able to simply detect
            // the deletion of a callsite, but I don't think there is any way to do so.
            // Instead every callsite has a generation counter and doing an inlining into a caller bumps the generation
            // counter by one. All callsites of an older generation are considered out of date and simply popped.
            containedCallable->walk(
                [&](mlir::CallOpInterface calls)
                {
                    seenCalls.erase(calls);
                    auto iter = callees.find(calls.resolveCallable(&collection));
                    if (iter == callees.end())
                    {
                        return;
                    }
                    iter->second.erase(calls);
                });

            callableInlinedCallables[containedCallable][callSite.callableOp]++;

            pylir::Py::inlineCall(callSite.callOp, callSite.callableOp);
            if (mlir::failed(mlir::applyPatternsAndFoldGreedily(containedCallable, patterns)))
            {
                signalPassFailure();
                return;
            }

            info.generation++;

            getAnalysisManager().nest(containedCallable).invalidate({});
            info.currentSize = getChildAnalysis<pylir::BodySize>(containedCallable).getSize();
            auto& invalid = callees[containedCallable];
            invalidatedCalls.insert(invalid.begin(), invalid.end());
            addCallOps(
                containedCallable, info.generation,
                [&](mlir::CallOpInterface, mlir::CallableOpInterface callableOpInterface) -> bool
                { return callableInlinedCallables[containedCallable][callableOpInterface] < m_maxRecursiveInlines; });
            m_callsInlined++;
        }
    }

    std::int32_t costBenefitHeuristic(mlir::CallOpInterface callOpInterface,
                                      mlir::CallableOpInterface callableOpInterface)
    {
        std::int32_t cost = getChildAnalysis<pylir::BodySize>(callableOpInterface).getSize();
        cost -= 1;                                       // Call instruction
        cost -= callOpInterface.getArgOperands().size(); // Call Arguments
        return cost;
    }
};

} // namespace

std::unique_ptr<mlir::Pass> pylir::Py::createInlinerPass()
{
    return std::make_unique<Inliner>();
}
