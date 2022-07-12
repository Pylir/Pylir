// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/Threading.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/ADT/DenseMap.h>

#include <pylir/Optimizer/Analysis/BodySize.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Util/InlinerUtil.hpp>
#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Variant.hpp>

#include <future>
#include <mutex>
#include <variant>

#include "PassDetail.hpp"
#include "Passes.hpp"

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
    llvm::DenseMap<CallingContext, std::shared_future<bool>> m_decisions;
    mutable std::mutex mutex;

public:
    std::variant<std::promise<bool>, bool> lookup(CallingContext context)
    {
        std::unique_lock lock{mutex};
        auto [iter, inserted] = m_decisions.try_emplace(std::move(context), std::shared_future<bool>{});
        if (!inserted)
        {
            auto copy = iter->second;
            lock.unlock();
            return copy.get();
        }
        std::promise<bool> promise;
        iter->second = promise.get_future().share();
        return promise;
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

class TrialInliner : public pylir::Py::TrialInlinerBase<TrialInliner>
{
    mlir::OpPassManager m_passManager;

    mlir::FailureOr<bool> performTrial(mlir::FunctionOpInterface functionOpInterface,
                                       mlir::CallOpInterface callOpInterface,
                                       mlir::CallableOpInterface callableOpInterface, std::size_t calleeSize,
                                       mlir::OpPassManager& passManager)
    {
        mlir::OwningOpRef<mlir::FunctionOpInterface> rollback = functionOpInterface.clone();
        auto callerSize = pylir::BodySize(functionOpInterface).getSize();
        pylir::Py::inlineCall(callOpInterface, callableOpInterface);

        if (mlir::failed(runPipeline(passManager, functionOpInterface)))
        {
            return mlir::failure();
        }

        auto newCombinedSize = pylir::BodySize(functionOpInterface).getSize();
        auto delta =
            static_cast<std::ptrdiff_t>(callerSize + calleeSize) - static_cast<std::ptrdiff_t>(newCombinedSize);
        auto reduction = 1.0 - (static_cast<std::ptrdiff_t>(calleeSize) - delta) / static_cast<double>(calleeSize);
        if (reduction >= m_minCalleeSizeReduction / 100.0)
        {
            m_callsInlined++;
            return true;
        }
        functionOpInterface.getBody().takeBody(rollback->getBody());
        return false;
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

    mlir::FailureOr<mlir::FunctionOpInterface> optimize(mlir::FunctionOpInterface functionOpInterface,
                                                        TrialDataBase& dataBase,
                                                        const llvm::DenseMap<mlir::StringAttr, Inlineable>& symbolTable,
                                                        mlir::OpPassManager& passManager)
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
                    auto ref = callOpInterface.getCallableForCallee()
                                   .dyn_cast<mlir::SymbolRefAttr>()
                                   .dyn_cast_or_null<mlir::FlatSymbolRefAttr>();
                    if (!ref)
                    {
                        return mlir::WalkResult::advance();
                    }
                    auto inlineable = symbolTable.find(ref.getAttr());
                    if (inlineable == symbolTable.end())
                    {
                        return mlir::WalkResult::advance();
                    }
                    if (disabledCallables.contains(inlineable->second.getCallable()))
                    {
                        return mlir::WalkResult::advance();
                    }
                    auto result = dataBase.lookup({ref.getAttr(), callOpInterface.getArgOperands()});
                    if (auto* maybeBool = std::get_if<bool>(&result))
                    {
                        m_cacheHits++;
                        if (!*maybeBool)
                        {
                            return mlir::WalkResult::advance();
                        }
                        recursivePattern(inlineable->second.getCallable());
                        m_callsInlined++;
                        pylir::Py::inlineCall(callOpInterface, inlineable->second.getCallable());
                        if (mlir::failed(runPipeline(passManager, functionOpInterface)))
                        {
                            failed = true;
                        }
                        return mlir::WalkResult::interrupt();
                    }
                    m_cacheMisses++;
                    auto trialResult =
                        performTrial(functionOpInterface, callOpInterface, inlineable->second.getCallable(),
                                     inlineable->second.getCalleeSize(), passManager);
                    if (mlir::failed(trialResult))
                    {
                        failed = true;
                        // Unblock other threads
                        pylir::get<std::promise<bool>>(result).set_value(false);
                        return mlir::WalkResult::interrupt();
                    }
                    pylir::get<std::promise<bool>>(result).set_value(*trialResult);
                    if (*trialResult)
                    {
                        // Add it to patterns afterwards. This is only really to add a new state machine for the
                        // callable. Any real patterns will be executed via cache hits. Since this was a cache miss
                        // it can't really have been part of any pattern and it doesn't matter that it will reset
                        // some of the existing state machines
                        recursivePattern(inlineable->second.getCallable());
                    }
                    // We have to interrupt regardless of whether it was rolled back or not as the specific blocks
                    // operations that make up the function body had all been copied and then moved, aka they are not
                    // the same anymore as in the walk operation. We might want to improve this situation in the future.
                    return mlir::WalkResult::interrupt();
                });
        } while (walkResult.wasInterrupted());
        if (failed)
        {
            return mlir::failure();
        }
        return functionOpInterface;
    }

protected:
    void runOnOperation() override
    {
        auto functions =
            llvm::to_vector(llvm::make_filter_range(getOperation().getOps<mlir::FunctionOpInterface>(),
                                                    [](mlir::FunctionOpInterface op) { return !op.isExternal(); }));
        llvm::DenseMap<mlir::StringAttr, Inlineable> originalCallables;
        for (auto iter : functions)
        {
            originalCallables.try_emplace(mlir::cast<mlir::SymbolOpInterface>(*iter).getNameAttr(), iter.clone(),
                                          getChildAnalysis<pylir::BodySize>(iter).getSize());
        }

        TrialDataBase dataBase;
        if (mlir::failed(mlir::failableParallelForEach(&getContext(), llvm::enumerate(functions),
                                                       [&](const auto& iter)
                                                       {
                                                           mlir::OpPassManager copy = m_passManager;
                                                           return optimize(iter.value(), dataBase, originalCallables,
                                                                           copy);
                                                       })))
        {
            signalPassFailure();
            return;
        }
    }

    mlir::LogicalResult initialize(mlir::MLIRContext*) override
    {
        return mlir::parsePassPipeline(m_optimizationPipeline, m_passManager);
    }
};
} // namespace

std::unique_ptr<mlir::Pass> pylir::Py::createTrialInlinerPass()
{
    return std::make_unique<TrialInliner>();
}
