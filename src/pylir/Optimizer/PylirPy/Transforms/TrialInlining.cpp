// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Analysis/CallGraph.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SCCIterator.h>

#include <pylir/Optimizer/Analysis/BodySize.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Util/InlinerUtil.hpp>
#include <pylir/Support/Macros.hpp>

#include "Passes.hpp"

#define DEBUG_TYPE "trial-inliner"

#include <llvm/Support/Debug.h>

namespace pylir::Py
{
#define GEN_PASS_DEF_TRIALINLINERPASS
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"
} // namespace pylir::Py

namespace
{
struct CallingContext
{
    mlir::StringAttr callee;
    std::vector<llvm::Optional<std::pair<mlir::OperationName, std::size_t>>> callArguments;

    explicit CallingContext(mlir::StringAttr callee) : callee(callee) {}

    CallingContext(mlir::StringAttr callee, mlir::OperandRange values) : callee(callee), callArguments(values.size())
    {
        for (auto [src, dest] : llvm::zip(values, callArguments))
        {
            if (auto res = src.dyn_cast<mlir::OpResult>())
            {
                dest.emplace(res.getOwner()->getName(), res.getResultNumber());
            }
        }
    }

    bool operator==(const CallingContext& rhs) const
    {
        return std::tie(callee, callArguments) == std::tie(rhs.callee, rhs.callArguments);
    }

    bool operator!=(const CallingContext& rhs) const
    {
        return !(rhs == *this);
    }
};

} // namespace

template <>
struct llvm::DenseMapInfo<CallingContext>
{
    static inline CallingContext getEmptyKey()
    {
        return CallingContext{llvm::DenseMapInfo<mlir::StringAttr>::getEmptyKey()};
    }

    static inline CallingContext getTombstoneKey()
    {
        return CallingContext{llvm::DenseMapInfo<mlir::StringAttr>::getTombstoneKey()};
    }

    static inline unsigned getHashValue(const CallingContext& value)
    {
        return llvm::hash_combine(value.callee,
                                  llvm::hash_combine_range(value.callArguments.begin(), value.callArguments.end()));
    }

    static inline bool isEqual(const CallingContext& lhs, const CallingContext& rhs)
    {
        return lhs == rhs;
    }
};

namespace
{

class TrialDataBase
{
public:
    enum class Result
    {
        NotComputed,
        Profitable,
        NotProfitable
    };

private:
    llvm::DenseMap<CallingContext, Result> m_decisions;

public:
    Result& lookup(CallingContext context)
    {
        return m_decisions.try_emplace(std::move(context), Result::NotComputed).first->second;
    }
};

class RecursionStateMachine
{
    std::vector<mlir::CallableOpInterface> m_pattern;
    decltype(m_pattern)::const_iterator m_inPattern;
    std::size_t m_count = 1;

    enum class States
    {
        GatheringPattern,
        Synchronizing,
        CheckingPattern,
    };

    States m_state = States::GatheringPattern;

public:
    explicit RecursionStateMachine(mlir::CallableOpInterface firstOccurrence) : m_pattern{firstOccurrence} {}

    ~RecursionStateMachine() = default;
    RecursionStateMachine(const RecursionStateMachine&) = delete;
    RecursionStateMachine& operator=(const RecursionStateMachine&) = delete;
    RecursionStateMachine(RecursionStateMachine&&) noexcept = default;
    RecursionStateMachine& operator=(RecursionStateMachine&&) noexcept = default;

    [[nodiscard]] std::size_t getCount() const
    {
        return m_count;
    }

    [[nodiscard]] llvm::ArrayRef<mlir::CallableOpInterface> getPattern() const
    {
        return m_pattern;
    }

    void cycle(mlir::CallableOpInterface occurrence)
    {
        switch (m_state)
        {
            case States::Synchronizing:
                if (occurrence != m_pattern.front())
                {
                    return;
                }
                m_state = States::GatheringPattern;
                break;
            case States::GatheringPattern:
                if (occurrence != m_pattern.front())
                {
                    m_pattern.push_back(occurrence);
                    return;
                }
                m_state = States::CheckingPattern;
                m_inPattern = m_pattern.begin() + (m_pattern.size() == 1 ? 0 : 1);
                m_count = 2;
                break;
            case States::CheckingPattern:
                if (*m_inPattern == occurrence)
                {
                    if (++m_inPattern == m_pattern.end())
                    {
                        m_count++;
                        m_inPattern = m_pattern.begin();
                    }
                    return;
                }
                m_state = occurrence == m_pattern.front() ? States::GatheringPattern : States::Synchronizing;
                m_count = 0;
                m_pattern.resize(1);
                break;
        }
    }
};

class TrialInliner : public pylir::Py::impl::TrialInlinerPassBase<TrialInliner>
{
    mlir::OpPassManager m_passManager;

    mlir::FailureOr<TrialDataBase::Result> performTrial(mlir::FunctionOpInterface functionOpInterface,
                                                        mlir::CallOpInterface callOpInterface,
                                                        mlir::CallableOpInterface callableOpInterface,
                                                        std::size_t calleeSize, mlir::OpPassManager& passManager)
    {
        mlir::OwningOpRef<mlir::FunctionOpInterface> rollback = functionOpInterface.clone();
        auto callerSize = pylir::BodySize(functionOpInterface).getSize();
        pylir::Py::inlineCall(callOpInterface, callableOpInterface);

        m_optimizationRun++;
        if (mlir::failed(runPipeline(passManager, functionOpInterface)))
        {
            return mlir::failure();
        }

        auto newCombinedSize = pylir::BodySize(functionOpInterface).getSize();
        auto delta =
            static_cast<std::ptrdiff_t>(callerSize + calleeSize) - static_cast<std::ptrdiff_t>(newCombinedSize);
        auto reduction = 1.0 - (static_cast<std::ptrdiff_t>(calleeSize) - delta) / static_cast<double>(calleeSize);
        LLVM_DEBUG({
            llvm::dbgs() << "Trial result: Reduction of " << static_cast<std::ptrdiff_t>(reduction * 100)
                         << "%. Required size reduction: " << m_minCalleeSizeReduction
                         << "%. Success = " << (reduction >= m_minCalleeSizeReduction / 100.0) << "\n";
        });
        if (reduction >= m_minCalleeSizeReduction / 100.0)
        {
            m_callsInlined++;
            return TrialDataBase::Result::Profitable;
        }
        functionOpInterface.getBody().takeBody(rollback->getBody());
        return TrialDataBase::Result::NotProfitable;
    }

    class Inlineable
    {
        mlir::OwningOpRef<mlir::FunctionOpInterface> m_functionOp;
        std::size_t m_calleeSize;

    public:
        Inlineable(mlir::OwningOpRef<mlir::FunctionOpInterface>&& functionOp, std::size_t calleeSize)
            : m_functionOp(std::move(functionOp)), m_calleeSize(calleeSize)
        {
        }

        [[nodiscard]] mlir::CallableOpInterface getCallable() const
        {
            return mlir::cast<mlir::CallableOpInterface>(**m_functionOp);
        }

        [[nodiscard]] std::size_t getCalleeSize() const
        {
            return m_calleeSize;
        }
    };

    mlir::LogicalResult optimize(mlir::FunctionOpInterface functionOpInterface, TrialDataBase& dataBase,
                                 mlir::SymbolTableCollection& symbolTable, mlir::OpPassManager& passManager)
    {
        llvm::MapVector<mlir::CallableOpInterface, RecursionStateMachine> recursionDetection;
        llvm::DenseSet<mlir::CallableOpInterface> disabledCallables;
        auto recursivePattern = [&](mlir::CallableOpInterface callable)
        {
            bool triggered = false;
            for (auto iter = recursionDetection.begin(); iter != recursionDetection.end();)
            {
                auto& stateMachine = iter->second;
                stateMachine.cycle(callable);
                if (stateMachine.getCount() >= m_maxRecursionDepth)
                {
                    auto pattern = stateMachine.getPattern();
                    disabledCallables.insert(pattern.begin(), pattern.end());
                    iter = recursionDetection.erase(iter);
                    triggered = true;
                    m_recursionLimitReached++;
                    continue;
                }
                iter++;
            }
            if (!triggered)
            {
                // If it was triggered, then this callable was the fist occurrence in a recursion pattern and now
                // disabled and deleted. Don't reinsert it now
                recursionDetection.insert({callable, RecursionStateMachine{callable}});
            }
        };

        bool failed = false;
        mlir::WalkResult walkResult(mlir::failure());
        do
        {
            walkResult = functionOpInterface->walk<mlir::WalkOrder::PreOrder>(
                [&](mlir::CallOpInterface callOpInterface)
                {
                    auto callable = mlir::dyn_cast_or_null<mlir::CallableOpInterface>(
                        callOpInterface.resolveCallable(&symbolTable));
                    if (!callable || !callable.getCallableRegion())
                    {
                        return mlir::WalkResult::advance();
                    }
                    if (disabledCallables.contains(callable))
                    {
                        return mlir::WalkResult::advance();
                    }
                    TrialDataBase::Result& result =
                        dataBase.lookup({mlir::cast<mlir::SymbolOpInterface>(*callable).getNameAttr(),
                                         callOpInterface.getArgOperands()});
                    if (result != TrialDataBase::Result::NotComputed)
                    {
                        m_cacheHits++;
                        if (result == TrialDataBase::Result::NotProfitable)
                        {
                            return mlir::WalkResult::advance();
                        }
                        recursivePattern(callable);
                        m_callsInlined++;
                        LLVM_DEBUG({
                            llvm::dbgs() << "Inlining " << mlir::cast<mlir::SymbolOpInterface>(*callable).getNameAttr()
                                         << " into " << functionOpInterface.getName() << '\n';
                        });
                        pylir::Py::inlineCall(callOpInterface, callable);
                        m_optimizationRun++;
                        if (mlir::failed(runPipeline(passManager, functionOpInterface)))
                        {
                            failed = true;
                        }
                        return mlir::WalkResult::interrupt();
                    }
                    m_cacheMisses++;
                    LLVM_DEBUG({
                        llvm::dbgs() << "Performing Trial of "
                                     << mlir::cast<mlir::SymbolOpInterface>(*callable).getNameAttr() << " into "
                                     << functionOpInterface.getName() << '\n';
                    });
                    auto trialResult = performTrial(functionOpInterface, callOpInterface, callable,
                                                    getChildAnalysis<pylir::BodySize>(callable).getSize(), passManager);
                    if (mlir::failed(trialResult))
                    {
                        failed = true;
                        return mlir::WalkResult::interrupt();
                    }
                    result = *trialResult;
                    if (*trialResult == TrialDataBase::Result::Profitable)
                    {
                        // Add it to patterns afterwards. This is only really to add a new state machine for the
                        // callable. Any real patterns will be executed via cache hits. Since this was a cache miss
                        // it can't really have been part of any pattern and it doesn't matter that it will reset
                        // some of the existing state machines
                        recursivePattern(callable);
                    }
                    // We have to interrupt regardless of whether it was rolled back or not as the specific blocks
                    // operations that make up the function body had all been copied and then moved, aka they are not
                    // the same anymore as in the walk operation. We might want to improve this situation in the future.
                    return mlir::WalkResult::interrupt();
                });
        } while (walkResult.wasInterrupted());
        return mlir::failure(failed);
    }

protected:
    void runOnOperation() override
    {
        // Run the optimization pipeline once beforehand to get a correct measurement on what improvements the inlining
        // would bring.
        mlir::OpPassManager passManager(getOperation()->getName());
        passManager.nestAny() = m_passManager;
        if (mlir::failed(runPipeline(passManager, getOperation())))
        {
            signalPassFailure();
            return;
        }

        TrialDataBase dataBase;
        mlir::SymbolTableCollection collection;
        auto& callgraph = getAnalysis<mlir::CallGraph>();
        for (auto iter = llvm::scc_begin(&std::as_const(callgraph)); !iter.isAtEnd(); ++iter)
        {
            LLVM_DEBUG({
                llvm::dbgs() << "Inlining SCC: ";
                llvm::interleaveComma(*iter, llvm::dbgs(),
                                      [](const mlir::CallGraphNode* node)
                                      {
                                          if (node->isExternal())
                                          {
                                              return;
                                          }
                                          llvm::dbgs() << mlir::cast<mlir::SymbolOpInterface>(
                                                              node->getCallableRegion()->getParentOp())
                                                              .getNameAttr();
                                      });
                llvm::dbgs() << '\n';
            });
            for (const auto& node : *iter)
            {
                if (node->isExternal())
                {
                    continue;
                }

                if (mlir::failed(
                        optimize(node->getCallableRegion()->getParentOp(), dataBase, collection, m_passManager)))
                {
                    signalPassFailure();
                    return;
                }

                node->getCallableRegion()->walk(
                    [&](mlir::CallOpInterface calls)
                    {
                        auto callable =
                            mlir::dyn_cast_or_null<mlir::CallableOpInterface>(calls.resolveCallable(&collection));
                        if (!callable || !callable.getCallableRegion())
                        {
                            return;
                        }
                        node->addCallEdge(callgraph.getOrAddNode(callable.getCallableRegion(), nullptr));
                    });
            }
        }
    }

    mlir::LogicalResult initialize(mlir::MLIRContext*) override
    {
        return mlir::parsePassPipeline(m_optimizationPipeline, m_passManager);
    }

public:
    using Base::Base;

    void getDependentDialects(mlir::DialectRegistry& registry) const override
    {
        Base::getDependentDialects(registry);
        // Above initialize will signal the error properly. This also gets called before `initialize`, hence we can't
        // use m_passManager here.
        mlir::OpPassManager temp;
        if (mlir::failed(mlir::parsePassPipeline(m_optimizationPipeline, temp, llvm::nulls())))
        {
            return;
        }
        temp.getDependentDialects(registry);
    }
};
} // namespace
