// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Analysis/Liveness.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Dominance.h>

#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ThreadPool.h>

#include <pylir/Optimizer/Analysis/LoopInfo.hpp>
#include <pylir/Optimizer/PylirPy/Analysis/TypeFlow.hpp>
#include <pylir/Optimizer/PylirPy/Analysis/TypeFlowInterfaces.hpp>
#include <pylir/Optimizer/PylirPy/IR/ObjectAttrInterface.hpp>
#include <pylir/Optimizer/PylirPy/IR/ObjectTypeInterface.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyAttributes.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Variant.hpp>

#include <mutex>
#include <queue>
#include <unordered_map>
#include <utility>
#include <variant>

#include "PassDetail.hpp"
#include "Passes.hpp"

namespace
{
class Monomorph : public pylir::Py::MonomorphBase<Monomorph>
{
protected:
    void runOnOperation() override;
};

/// The Lattice of the Dataflow analysis. It's top value is the NULL value. The bottom value is not directly represented
/// in the lattice, but is indicated by the absence of a mapped TypeFlowValue in the respective maps.
class TypeFlowValue : public mlir::Attribute
{
public:
    TypeFlowValue() = default;

    /*implicit*/ TypeFlowValue(pylir::Py::ObjectTypeInterface type) : mlir::Attribute(mlir::TypeAttr::get(type)) {}

    /*implicit*/ TypeFlowValue(mlir::Attribute attr) : mlir::Attribute(attr) {}

    TypeFlowValue join(TypeFlowValue rhs)
    {
        if (!rhs || !*this)
        {
            return {};
        }
        if (*this == rhs)
        {
            return *this;
        }
        if (auto thisType = dyn_cast<mlir::TypeAttr>())
        {
            if (auto rhsType = rhs.dyn_cast<mlir::TypeAttr>())
            {
                return pylir::Py::joinTypes(thisType.getValue(), rhsType.getValue());
            }
        }
        return {};
    }
};

using TypeFlowArgValue = std::variant<mlir::FlatSymbolRefAttr, pylir::Py::ObjectTypeInterface>;

class FunctionSpecialization
{
    mlir::Operation* m_function;
    std::vector<TypeFlowArgValue> m_argTypes;

    friend struct llvm::DenseMapInfo<FunctionSpecialization>;

    FunctionSpecialization(mlir::Operation* function) : m_function(function) {}

public:
    FunctionSpecialization(mlir::FunctionOpInterface function, std::vector<TypeFlowArgValue> argTypes)
        : m_function(function), m_argTypes(std::move(argTypes))
    {
    }

    [[nodiscard]] mlir::FunctionOpInterface getFunction() const
    {
        return m_function;
    }

    llvm::ArrayRef<TypeFlowArgValue> getArgTypes() const
    {
        return m_argTypes;
    }

    bool operator==(const FunctionSpecialization& rhs) const
    {
        return std::tie(m_function, m_argTypes) == std::tie(rhs.m_function, rhs.m_argTypes);
    }

    bool operator!=(const FunctionSpecialization& rhs) const
    {
        return !(rhs == *this);
    }
};

} // namespace

template <>
struct llvm::DenseMapInfo<FunctionSpecialization>
{
    static inline FunctionSpecialization getEmptyKey()
    {
        return {llvm::DenseMapInfo<mlir::Operation*>::getEmptyKey()};
    }

    static inline FunctionSpecialization getTombstoneKey()
    {
        return {llvm::DenseMapInfo<mlir::Operation*>::getTombstoneKey()};
    }

    static inline unsigned getHashValue(const FunctionSpecialization& value)
    {
        auto f = [](auto unionVal)
        { return (std::uintptr_t)pylir::match(unionVal, [](auto&& arg) { return arg.getAsOpaquePointer(); }); };
        auto argTypes = value.getArgTypes();
        return llvm::hash_combine(&*value.getFunction(),
                                  llvm::hash_combine_range(llvm::mapped_iterator(argTypes.begin(), f),
                                                           llvm::mapped_iterator(argTypes.end(), f)));
    }

    static inline bool isEqual(const FunctionSpecialization& lhs, const FunctionSpecialization& rhs)
    {
        return lhs == rhs;
    }
};

template <>
struct llvm::DenseMapInfo<std::vector<TypeFlowValue>>
{
    static inline std::vector<TypeFlowValue> getEmptyKey()
    {
        return {llvm::DenseMapInfo<TypeFlowValue>::getEmptyKey()};
    }

    static inline std::vector<TypeFlowValue> getTombstoneKey()
    {
        return {llvm::DenseMapInfo<TypeFlowValue>::getTombstoneKey()};
    }

    static inline unsigned getHashValue(const std::vector<TypeFlowValue>& value)
    {
        return llvm::hash_combine_range(value.begin(), value.end());
    }

    static inline bool isEqual(const std::vector<TypeFlowValue>& lhs, const std::vector<TypeFlowValue>& rhs)
    {
        return lhs == rhs;
    }
};

namespace
{

struct FunctionCall
{
    FunctionSpecialization functionSpecialization;
    mlir::ValueRange resultValues;
    mlir::Operation* callOp;
};

struct SuccessorBlocks
{
    mlir::SuccessorRange successors{};
    mlir::Block* skipped = nullptr;
};

using ExecutionResult = std::variant<SuccessorBlocks, FunctionCall, llvm::SmallVector<pylir::Py::ObjectTypeInterface>>;

class ExecutionFrame
{
    /// Op that is about to be executed.
    mlir::Operation* m_nextExecutedOp;
    /// Mapping of TypeFlowIR meta values and their actual current values.
    llvm::DenseMap<mlir::Value, TypeFlowValue> m_values;

public:
    explicit ExecutionFrame(mlir::Operation* nextExecutedOp,
                            llvm::DenseMap<mlir::Value, TypeFlowValue> valuesInit = decltype(m_values)(0))
        : m_nextExecutedOp(nextExecutedOp), m_values(std::move(valuesInit))
    {
    }

    [[nodiscard]] mlir::Operation& getNextExecutedOp() const
    {
        return *m_nextExecutedOp;
    }

    [[nodiscard]] llvm::DenseMap<mlir::Value, TypeFlowValue>& getValues()
    {
        return m_values;
    }

    /// Attempts to execute as many operations as possible. Execution may return if either a terminator or a
    /// a call has been encountered.
    /// If a terminator was the cause of returning, a SuccessorRange with 0 or more
    /// successors is returned, containing the successor blocks that should be executed next. The Frame may not be
    /// executed again.
    /// If a call was the cause of returning, a FunctionCall containing the callee and arguments is returned.
    /// Execution may be resumed at the operation after the call by simply calling execute again.
    /// If a return instruction was the cause of returning, a SmallVector containing the return values is returned.
    /// The Frame may not be executed again.
    ExecutionResult execute(mlir::SymbolTableCollection& collection)
    {
        PYLIR_ASSERT(m_nextExecutedOp);
        while (true)
        {
            auto* currOp = m_nextExecutedOp;
            m_nextExecutedOp = m_nextExecutedOp->getNextNode();
            std::optional<ExecutionResult> optional =
                llvm::TypeSwitch<mlir::Operation*, std::optional<ExecutionResult>>(currOp)
                    .Case(
                        [&](pylir::TypeFlow::ConstantOp constant)
                        {
                            m_values[constant] = constant.getInput();
                            return std::nullopt;
                        })
                    .Case(
                        [&](pylir::TypeFlow::UndefOp undef)
                        {
                            m_values[undef] = {};
                            return std::nullopt;
                        })
                    .Case(
                        [&](pylir::TypeFlow::TypeFlowExecInterface interface)
                        {
                            llvm::SmallVector<mlir::OpFoldResult> result;
                            if (mlir::failed(interface.exec(
                                    llvm::to_vector(llvm::map_range(interface->getOperands(),
                                                                    [&](mlir::Value val) -> mlir::Attribute
                                                                    { return m_values[val]; })),
                                    result, collection)))
                            {
                                for (auto iter : interface->getOpResults())
                                {
                                    m_values[iter] = {};
                                }
                            }
                            else
                            {
                                PYLIR_ASSERT(interface->getNumResults() == result.size());
                                for (auto [res, value] : llvm::zip(interface->getOpResults(), result))
                                {
                                    if (auto attr = value.dyn_cast<mlir::Attribute>())
                                    {
                                        m_values[res] = attr;
                                    }
                                    else
                                    {
                                        m_values[res] = m_values[value.get<mlir::Value>()];
                                    }
                                }
                            }
                            return std::nullopt;
                        })
                    .Case(
                        [&](pylir::TypeFlow::BranchOp branchOp) {
                            return SuccessorBlocks{branchOp.getSuccessors(), nullptr};
                        })
                    .Case(
                        [&](pylir::TypeFlow::CondBranchOp condBranchOp)
                        {
                            auto cond = m_values[condBranchOp.getCondition()];
                            if (auto boolean = cond.dyn_cast_or_null<mlir::BoolAttr>())
                            {
                                return SuccessorBlocks{
                                    boolean.getValue() ? condBranchOp.getTrueSucc() : condBranchOp.getFalseSucc(),
                                    boolean.getValue() ? condBranchOp.getFalseSucc() : condBranchOp.getTrueSucc(),
                                };
                            }
                            return SuccessorBlocks{condBranchOp.getSuccessors(), nullptr};
                        })
                    .Case(
                        [&](pylir::TypeFlow::ReturnOp returnOp)
                        {
                            llvm::SmallVector<pylir::Py::ObjectTypeInterface> returnedValues(
                                returnOp.getValues().size());
                            for (auto [returnValue, value] : llvm::zip(returnedValues, returnOp.getValues()))
                            {
                                if (auto typeAttr = m_values[value].dyn_cast_or_null<mlir::TypeAttr>())
                                {
                                    returnValue = typeAttr.getValue();
                                }
                            }
                            return returnedValues;
                        })
                    .Case<pylir::TypeFlow::CallOp, pylir::TypeFlow::CallIndirectOp>(
                        [&](auto callOp) -> std::optional<ExecutionResult>
                        {
                            mlir::FunctionOpInterface function;
                            if constexpr (std::is_same_v<std::decay_t<decltype(callOp)>, pylir::TypeFlow::CallOp>)
                            {
                                function = collection.lookupNearestSymbolFrom<mlir::FunctionOpInterface>(
                                    callOp.getContext(), callOp.getCallee());
                            }
                            else
                            {
                                auto calleeValue = m_values[callOp.getCallee()];
                                auto funcAttr = calleeValue.template dyn_cast_or_null<pylir::Py::FunctionAttr>();
                                if (!funcAttr)
                                {
                                    if (auto callee = calleeValue.template dyn_cast_or_null<mlir::FlatSymbolRefAttr>())
                                    {
                                        auto functionObject =
                                            collection.lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(
                                                callOp.getContext(), callee);
                                        if (functionObject)
                                        {
                                            funcAttr = functionObject.getInitializerAttr()
                                                           .template dyn_cast_or_null<pylir::Py::FunctionAttr>();
                                        }
                                    }
                                }

                                if (!funcAttr)
                                {
                                    for (auto iter : callOp.getResults())
                                    {
                                        m_values[iter] = {};
                                    }
                                    return std::nullopt;
                                }
                                function = collection.lookupNearestSymbolFrom<mlir::FunctionOpInterface>(
                                    callOp.getContext(), funcAttr.getValue());
                            }

                            std::vector<TypeFlowArgValue> arguments(callOp.getArguments().size());
                            for (auto [argValue, inValue] : llvm::zip(arguments, callOp.getArguments()))
                            {
                                auto value = m_values[inValue];
                                // TODO: Restrict below that ref may only refer to a type
                                if (auto ref = value.template dyn_cast_or_null<mlir::FlatSymbolRefAttr>())
                                {
                                    argValue = ref;
                                }
                                else if (auto type = value.template dyn_cast_or_null<mlir::TypeAttr>())
                                {
                                    argValue = type.getValue();
                                }
                                else if (value.template isa_and_nonnull<pylir::Py::ObjectAttrInterface,
                                                                        mlir::SymbolRefAttr>())
                                {
                                    argValue = pylir::Py::typeOfConstant(value, collection, callOp.getContext());
                                }
                            }
                            return FunctionCall{FunctionSpecialization{function, std::move(arguments)},
                                                callOp.getResults(), callOp.getContext()};
                        });
            if (optional)
            {
                return *std::move(optional);
            }
        }
    }
};

struct TypeFlowInstance
{
    pylir::Py::TypeFlow& typeFLowIR;
    pylir::LoopInfo loopInfo;
    mlir::DominanceInfo dominanceInfo;
    mlir::Liveness liveness;

    TypeFlowInstance(pylir::Py::TypeFlow& typeFLowIr, pylir::LoopInfo&& loopInfo, mlir::DominanceInfo&& dominanceInfo,
                     mlir::Liveness&& liveness)
        : typeFLowIR(typeFLowIr),
          loopInfo(std::move(loopInfo)),
          dominanceInfo(std::move(dominanceInfo)),
          liveness(std::move(liveness))
    {
    }
};

/// Handles scheduling on the basic block level of a specific function call. It executes ExecutionFrames, which usually
/// execute a basic block, and then handles scheduling which basic blocks should be executed next.
class Orchestrator
{
    pylir::Py::TypeFlow& m_typeFLowIR;
    pylir::LoopInfo& m_loopInfo;
    mlir::DominanceInfo& m_dominanceInfo;
    mlir::Liveness& m_liveness;

    std::mutex m_orchestratorLock;
    std::vector<pylir::Py::ObjectTypeInterface> m_returnTypes; // protected by m_orchestratorLock
    llvm::DenseMap<mlir::Block*, bool> m_finishedBlocks;       // protected by m_orchestratorLock
    llvm::DenseMap<mlir::Value, TypeFlowValue> m_values;       // protected by m_orchestratorLock

    struct Loop
    {
        llvm::SetVector<mlir::Block*> exitBlocks;
        llvm::DenseSet<std::vector<TypeFlowValue>> headerArgs;

        Loop() : exitBlocks({}) {}
    };
    llvm::DenseMap<pylir::Loop*, Loop> m_loops; // protected by m_orchestratorLock
    std::mutex m_callSiteLock;
    llvm::DenseMap<mlir::Operation*, FunctionSpecialization> m_callSites; // protected by m_callSiteLock

    std::atomic_size_t m_inQueueCount = 0;

    std::vector<ExecutionFrame> buildSuccessorFrames(llvm::ArrayRef<std::pair<mlir::Block*, pylir::Loop*>> blocks)
    {
        std::vector<ExecutionFrame> results;
        for (auto [iter, loop] : blocks)
        {
            std::vector<TypeFlowValue> blockArgs(iter->getNumArguments());
            llvm::DenseMap<mlir::Value, TypeFlowValue> values;
            for (auto dominatingDef : m_liveness.getLiveIn(iter))
            {
                values[dominatingDef] = m_values[dominatingDef];
            }

            // Creating block args values from predecessor branch arguments
            for (auto pred = iter->pred_begin(); pred != iter->pred_end(); pred++)
            {
                if (!m_finishedBlocks.lookup(*pred))
                {
                    continue;
                }
                auto branchOp = mlir::dyn_cast_or_null<mlir::BranchOpInterface>((*pred)->getTerminator());
                if (!branchOp)
                {
                    for (auto arg : iter->getArguments())
                    {
                        values[arg] = {};
                        blockArgs[arg.getArgNumber()] = {};
                    }
                    continue;
                }

                auto succArgs = branchOp.getSuccessorOperands(pred.getSuccessorIndex());
                for (const auto& arg : iter->getArguments())
                {
                    auto succArg = succArgs[arg.getArgNumber()];
                    if (!succArg)
                    {
                        values[arg] = {};
                        blockArgs[arg.getArgNumber()] = {};
                        continue;
                    }
                    auto succValue = m_values[succArg];
                    auto [existing, inserted] = values.insert({arg, succValue});
                    if (!inserted)
                    {
                        existing->second = existing->second.join(succValue);
                    }
                    blockArgs[arg.getArgNumber()] = existing->second;
                }
            }
            if (!loop)
            {
                results.emplace_back(&iter->front(), std::move(values));
                continue;
            }

            auto& loopOrch = m_loops[loop];
            // TODO: Less lazy and memory inefficient that is not just saving the block arguments but checking for
            //       changes.
            if (loopOrch.headerArgs.insert(std::move(blockArgs)).second)
            {
                // New loop iteration. Remove all loop blocks from finishedBlocks to make them reevaluated properly.
                for (auto* block : loop->getBlocks())
                {
                    m_finishedBlocks.erase(block);
                }
                results.emplace_back(&iter->front(), std::move(values));
                continue;
            }
            // We have previously encountered this loop with these block args and hence reached a fixpoint.
            // Unleash the exits!
            auto temp = handleNoAndIntoLoopSuccessors(loopOrch.exitBlocks.getArrayRef());
            results.insert(results.end(), std::move_iterator(temp.begin()), std::move_iterator(temp.end()));
            m_loops.erase(loop);
        }
        return results;
    }

    std::vector<ExecutionFrame> handleNoAndIntoLoopSuccessors(mlir::BlockRange successors)
    {
        llvm::SmallVector<std::pair<mlir::Block*, pylir::Loop*>> successor;
        for (auto* iter : successors)
        {
            // If all predecessors results are ready then, schedule this successor to be executed.
            // Special case if the successor is a loop header. In that case, back edges are ignored.
            // Since every loop entry of a natural loop is the loop header, if we are currently not in a loop,
            // but a successor block is, it is the loop header.
            auto* succLoop = m_loopInfo.getLoopFor(iter);
            if (llvm::all_of(iter->getPredecessors(), [this, succLoop](mlir::Block* pred)
                             { return m_finishedBlocks.count(pred) || (succLoop && succLoop->contains(pred)); }))
            {
                successor.emplace_back(iter, succLoop);
            }
        }
        return buildSuccessorFrames(successor);
    }

    friend class OrchestratorRefCount;

public:
    bool finishedExecution() const
    {
        return m_inQueueCount.load(std::memory_order_relaxed);
    }

    llvm::ArrayRef<pylir::Py::ObjectTypeInterface> getReturnTypes() const
    {
        return m_returnTypes;
    }

    const llvm::DenseMap<mlir::Value, TypeFlowValue>& getValues() const
    {
        return m_values;
    }

    const llvm::DenseMap<mlir::Operation*, FunctionSpecialization>& getCallSites() const
    {
        return m_callSites;
    }

    explicit Orchestrator(TypeFlowInstance& instance)
        : m_typeFLowIR(instance.typeFLowIR),
          m_loopInfo(instance.loopInfo),
          m_dominanceInfo(instance.dominanceInfo),
          m_liveness(instance.liveness),
          m_returnTypes(m_typeFLowIR.getFunction().getNumResults(),
                        pylir::Py::UnboundType::get(m_typeFLowIR.getFunction()->getContext()))
    {
    }

    [[nodiscard]] mlir::Block* getEntryBlock() const
    {
        return &m_typeFLowIR.getFunction().front();
    }

    /// Executes an ExecutionFrame. This may return either a vector of new Execution frames that should be executed
    /// or a FunctionCall. If it was a FunctionCall, then the executed execution frame was suspended and shall be
    /// executed again after the function call and it's result was computed.
    ///
    /// This function may be called from different threads.
    std::variant<std::vector<ExecutionFrame>, FunctionCall> execute(ExecutionFrame& executionFrame,
                                                                    mlir::SymbolTableCollection& symbolTableCollection)
    {
        mlir::Block* block = executionFrame.getNextExecutedOp().getBlock();
        auto result = executionFrame.execute(symbolTableCollection);
        if (auto* call = std::get_if<FunctionCall>(&result))
        {
            std::scoped_lock lock(m_callSiteLock);
            auto [existing, inserted] = m_callSites.insert({call->callOp, call->functionSpecialization});
            if (!inserted && existing->second != call->functionSpecialization)
            {
                existing->second = FunctionSpecialization{nullptr, {}};
            }
            return {std::move(*call)};
        }

        // Save this blocks result first of all.
        std::scoped_lock lock(m_orchestratorLock);
        m_finishedBlocks[block] = true;
        {
            auto& newValues = executionFrame.getValues();
            m_values.insert(newValues.begin(), newValues.end());
        }
        if (auto* vec = std::get_if<llvm::SmallVector<pylir::Py::ObjectTypeInterface>>(&result))
        {
            for (auto [res, arg] : llvm::zip(m_returnTypes, *vec))
            {
                res = pylir::Py::joinTypes(res, arg);
            }
            return {};
        }

        std::vector<ExecutionFrame> frames;
        auto& successorBlocks = pylir::get<SuccessorBlocks>(result);
        if (successorBlocks.skipped)
        {
            std::vector<mlir::DominanceInfoNode*> skippedSubTrees = {m_dominanceInfo.getNode(successorBlocks.skipped)};
            llvm::df_iterator_default_set<mlir::DominanceInfoNode*> visitedSet;
            while (!skippedSubTrees.empty())
            {
                auto* back = skippedSubTrees.back();
                skippedSubTrees.pop_back();
                for (auto* iter : llvm::depth_first_ext(back, visitedSet))
                {
                    m_finishedBlocks[block] = false;
                    // If we reached a leaf we need to check whether it's successor have to either be skipped as well
                    // or maybe even executed.
                    if (!iter->isLeaf())
                    {
                        continue;
                    }
                    llvm::SmallVector<std::pair<mlir::Block*, pylir::Loop*>> successors;
                    for (auto* succ : iter->getBlock()->getSuccessors())
                    {
                        if (m_finishedBlocks.count(succ))
                        {
                            continue;
                        }
                        auto* succLoop = m_loopInfo.getLoopFor(succ);
                        bool skipping = true;
                        bool ready = true;
                        for (auto* pred : succ->getPredecessors())
                        {
                            auto res = m_finishedBlocks.find(pred);
                            if (res != m_finishedBlocks.end())
                            {
                                skipping = skipping && !res->second;
                                continue;
                            }
                            if (succLoop && succLoop->contains(pred))
                            {
                                continue;
                            }
                            ready = false;
                            break;
                        }
                        if (!ready)
                        {
                            continue;
                        }
                        if (skipping)
                        {
                            skippedSubTrees.push_back(m_dominanceInfo.getNode(succ));
                            continue;
                        }
                        successors.emplace_back(succ, succLoop);
                    }
                    frames = buildSuccessorFrames(successors);
                }
            }
        }

        auto* loop = m_loopInfo.getLoopFor(block);
        if (!loop)
        {
            auto temp = handleNoAndIntoLoopSuccessors(successorBlocks.successors);
            frames.insert(frames.end(), std::move_iterator(temp.begin()), std::move_iterator(temp.end()));
            return frames;
        }

        auto& loopOrch = m_loops[loop];
        llvm::SmallVector<std::pair<mlir::Block*, pylir::Loop*>> successors;
        for (auto* succ : successorBlocks.successors)
        {
            if (loop->contains(succ))
            {
                auto* succLoop = m_loopInfo.getLoopFor(succ);
                successors.emplace_back(succ, succLoop->getHeader() == succ ? succLoop : nullptr);
                continue;
            }
            loopOrch.exitBlocks.insert(succ);
        }

        auto temp = buildSuccessorFrames(successors);
        frames.insert(frames.end(), std::move_iterator(temp.begin()), std::move_iterator(temp.end()));
        return frames;
    }
};

class OrchestratorRefCount
{
    Orchestrator* m_orchestrator = nullptr;

public:
    OrchestratorRefCount() = default;
    OrchestratorRefCount(std::nullptr_t) = delete;

    explicit OrchestratorRefCount(Orchestrator* orchestrator) : m_orchestrator(orchestrator)
    {
        m_orchestrator->m_inQueueCount.fetch_add(1, std::memory_order::memory_order_relaxed);
    }

    ~OrchestratorRefCount()
    {
        if (m_orchestrator)
        {
            m_orchestrator->m_inQueueCount.fetch_sub(1, std::memory_order::memory_order_relaxed);
        }
    }

    std::size_t release()
    {
        PYLIR_ASSERT(m_orchestrator);
        return std::exchange(m_orchestrator, nullptr)
                   ->m_inQueueCount.fetch_sub(1, std::memory_order::memory_order_relaxed)
               - 1;
    }

    OrchestratorRefCount(const OrchestratorRefCount&) = delete;
    OrchestratorRefCount& operator=(const OrchestratorRefCount&) = delete;

    OrchestratorRefCount(OrchestratorRefCount&& rhs) noexcept
        : m_orchestrator(std::exchange(rhs.m_orchestrator, nullptr))
    {
    }

    OrchestratorRefCount& operator=(OrchestratorRefCount&& rhs) noexcept
    {
        if (m_orchestrator)
        {
            m_orchestrator->m_inQueueCount.fetch_sub(std::memory_order::memory_order_relaxed);
        }
        m_orchestrator = std::exchange(rhs.m_orchestrator, nullptr);
        return *this;
    }

    Orchestrator& operator*() const
    {
        return *m_orchestrator;
    }

    Orchestrator* operator->() const
    {
        return m_orchestrator;
    }

    [[nodiscard]] Orchestrator* get() const
    {
        return m_orchestrator;
    }
};

/// Responsible for managing Orchestrators and their execution.
class Scheduler
{
    llvm::ThreadPool& m_threadPool;
    llvm::DenseMap<mlir::Operation*, std::unique_ptr<TypeFlowInstance>> m_typeFlowInstances;
    llvm::DenseMap<FunctionSpecialization, std::unique_ptr<Orchestrator>> m_orchestrators;

    struct CallWaiting
    {
        ExecutionFrame frame;
        OrchestratorRefCount orchestrator;
        mlir::ValueRange resultValues;
    };

    llvm::DenseMap<Orchestrator*, std::vector<CallWaiting>> m_callDependents;

    std::unique_ptr<Orchestrator> createOrchestrator(mlir::FunctionOpInterface function,
                                                     mlir::AnalysisManager moduleManager)
    {
        auto& typeFlowIR = moduleManager.getChildAnalysis<pylir::Py::TypeFlow>(function);
        mlir::ModuleAnalysisManager typeFlowModuleAnalysis(typeFlowIR.getFunction(), nullptr);
        mlir::AnalysisManager typeFlowAnalysisManager = typeFlowModuleAnalysis;
        auto& instance =
            m_typeFlowInstances
                .insert({function, std::make_unique<TypeFlowInstance>(
                                       typeFlowIR, std::move(typeFlowAnalysisManager.getAnalysis<pylir::LoopInfo>()),
                                       std::move(typeFlowAnalysisManager.getAnalysis<mlir::DominanceInfo>()),
                                       std::move(typeFlowAnalysisManager.getAnalysis<mlir::Liveness>()))})
                .first->second;
        return std::make_unique<Orchestrator>(*instance);
    }

public:
    explicit Scheduler(llvm::ThreadPool& threadPool) : m_threadPool(threadPool) {}

    /// Run the typeflow analysis starting from the given root functions. These may not take any DynamicType function
    /// arguments.
    void run(llvm::ArrayRef<mlir::FunctionOpInterface> roots, mlir::AnalysisManager moduleManager)
    {
        std::queue<std::pair<ExecutionFrame, OrchestratorRefCount>> queue;

        (void)m_threadPool;
        for (auto iter : roots)
        {
            auto spec = FunctionSpecialization(iter, {});
            auto function = spec.getFunction();
            auto& orchestrator = m_orchestrators[std::move(spec)] = createOrchestrator(function, moduleManager);
            queue.emplace(ExecutionFrame(&orchestrator->getEntryBlock()->front()), orchestrator.get());
        }

        mlir::SymbolTableCollection collection;
        while (!queue.empty())
        {
            auto front = std::move(queue.front());
            queue.pop();
            auto result = front.second->execute(front.first, collection);
            if (auto* vec = std::get_if<std::vector<ExecutionFrame>>(&result))
            {
                if (!vec->empty())
                {
                    // Successor blocks and this execution round was definitely not the last one from the orchestrator.
                    for (auto& iter : *vec)
                    {
                        queue.emplace(std::move(iter), std::move(front.second));
                    }
                    continue;
                }

                // Orchestrator returned without a successor and might have finished function execution of the whole
                // function.
                auto* orch = front.second.get();
                if (front.second.release() != 0)
                {
                    continue;
                }

                auto res = m_callDependents.find(orch);
                if (res == m_callDependents.end())
                {
                    continue;
                }

                for (auto& iter : res->second)
                {
                    for (auto [dest, value] : llvm::zip(iter.resultValues, orch->getReturnTypes()))
                    {
                        iter.frame.getValues()[dest] = value;
                    }
                    queue.emplace(std::move(iter.frame), std::move(iter.orchestrator));
                }

                continue;
            }
            auto& call = pylir::get<FunctionCall>(result);
            // Easy case of first time calling a function and no recursion
            auto [existing, inserted] = m_orchestrators.insert({std::move(call.functionSpecialization), nullptr});
            if (inserted)
            {
                existing->second = createOrchestrator(existing->first.getFunction(), moduleManager);
                llvm::DenseMap<mlir::Value, TypeFlowValue> entryValues;
                for (auto [arg, value] :
                     llvm::zip(existing->second->getEntryBlock()->getArguments(), existing->first.getArgTypes()))
                {
                    entryValues[arg] = pylir::match(
                        value,
                        [](pylir::Py::ObjectTypeInterface type) -> mlir::Attribute
                        { return mlir::TypeAttr::get(type); },
                        [](mlir::FlatSymbolRefAttr attr) -> mlir::Attribute { return attr; });
                }
                // No need to wait for the call if it has no results for us.
                if (call.resultValues.empty())
                {
                    queue.push(std::move(front));
                }
                else
                {
                    m_callDependents[existing->second.get()].push_back(
                        {std::move(front.first), std::move(front.second), call.resultValues});
                }
                queue.emplace(ExecutionFrame(&existing->second->getEntryBlock()->front(), std::move(entryValues)),
                              existing->second.get());

                continue;
            }
            // TODO: Recursion/not yet ready case
            [] {}();
        }
    }

    llvm::DenseMap<FunctionSpecialization, std::unique_ptr<Orchestrator>>& getResults()
    {
        return m_orchestrators;
    }
};

bool setCallee(mlir::Operation* op, mlir::FlatSymbolRefAttr callee)
{
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
        .Case<pylir::Py::CallOp, pylir::Py::InvokeOp>(
            [&](auto&& op)
            {
                if (op.getCalleeAttr() == callee)
                {
                    return false;
                }
                op.setCalleeAttr(callee);
                return true;
            })
        .Case(
            [&](pylir::Py::FunctionCallOp op)
            {
                // If it hasn't been turned into a constant than redirecting it to a clone of the given constant
                // is not valid either.
                if (!mlir::matchPattern(op.getFunction(), mlir::m_Constant()))
                {
                    return false;
                }
                mlir::OpBuilder builder(op);
                auto newCall =
                    builder.create<pylir::Py::CallOp>(op.getLoc(), op->getResultTypes(), callee, op.getCallOperands());
                newCall->setAttr(pylir::Py::alwaysBoundAttr, builder.getUnitAttr());
                op->replaceAllUsesWith(newCall);
                op->erase();
                return true;
            })
        .Case(
            [&](pylir::Py::FunctionInvokeOp op)
            {
                if (!mlir::matchPattern(op.getFunction(), mlir::m_Constant()))
                {
                    return false;
                }
                mlir::OpBuilder builder(op);
                auto newCall = builder.create<pylir::Py::InvokeOp>(
                    op.getLoc(), op->getResultTypes(), callee, op.getCallOperands(), op.getNormalDestOperands(),
                    op.getUnwindDestOperands(), op.getHappyPath(), op.getExceptionPath());
                newCall->setAttr(pylir::Py::alwaysBoundAttr, builder.getUnitAttr());
                op->replaceAllUsesWith(newCall);
                op->erase();
                return true;
            });
}

void Monomorph::runOnOperation()
{
    // TODO: Use SetVector once interfaces properly function in DenseMapInfo
    llvm::SmallVector<mlir::FunctionOpInterface> roots;
    llvm::SmallPtrSet<mlir::Operation*, 8> rootSet;
    for (auto iter : getOperation().getOps<mlir::FunctionOpInterface>())
    {
        if (llvm::none_of(iter.getArgumentTypes(), std::mem_fn(&mlir::Type::isa<pylir::Py::DynamicType>)))
        {
            roots.push_back(iter);
            rootSet.insert(iter);
        }
    }

    llvm::DenseMap<FunctionSpecialization, std::unique_ptr<Orchestrator>> results;
    {
        Scheduler scheduler(getContext().getThreadPool());
        scheduler.run(roots, getAnalysisManager());
        results = std::move(scheduler.getResults());
    }

    struct Clone
    {
        mlir::FunctionOpInterface function;
        mlir::BlockAndValueMapping mapping;
    };

    bool changed = false;
    mlir::SymbolTable table(getOperation());
    llvm::DenseMap<FunctionSpecialization, Clone> clones;
    for (auto& [func, orchestrator] : results)
    {
        auto& clone = clones[func];
        clone.function = func.getFunction();
        bool isRoot = rootSet.contains(func.getFunction());
        for (const auto& [key, value] : orchestrator->getValues())
        {
            if (!value || value.isa<mlir::TypeAttr>())
            {
                continue;
            }
            auto mapping = key.getDefiningOp<pylir::TypeFlow::TypeFlowValueMappingInterface>();
            if (!mapping)
            {
                continue;
            }

            // Cloning of a function body is done lazily for the case where no value has changed.
            // Roots are not cloned but updated in place.
            if (clone.function == func.getFunction() && !isRoot)
            {
                clone.function = clone.function->clone(clone.mapping);
                table.insert(clone.function);
                m_functionsCloned++;
            }

            auto instrValue = mapping.mapValue(key);
            auto cloneValue = clone.mapping.lookupOrDefault(instrValue);
            mlir::OpBuilder builder(cloneValue.getDefiningOp());
            auto* constant = cloneValue.getDefiningOp()->getDialect()->materializeConstant(
                builder, value, cloneValue.getType(), cloneValue.getLoc());
            PYLIR_ASSERT(constant);
            cloneValue.replaceAllUsesWith(constant->getResult(0));
            m_valuesReplaced++;
            changed = true;
        }
    }

    for (auto& orchestrator : llvm::make_second_range(results))
    {
        for (const auto& [origCall, func] : orchestrator->getCallSites())
        {
            if (!func.getFunction())
            {
                continue;
            }

            auto& mapping = clones[func].mapping;
            // TODO: Map call properly (not like the following code) as soon as it is supported by
            // BlockAndValueMapping
            mlir::Operation* call;
            if (origCall->getNumResults() != 0)
            {
                call = mapping.lookupOrDefault(origCall->getResult(0)).getDefiningOp();
            }
            else
            {
                auto* mappedBlock = mapping.lookupOrDefault(origCall->getBlock());
                auto distance = std::distance(origCall->getBlock()->begin(), mlir::Block::iterator{origCall});
                call = &*std::next(mappedBlock->begin(), distance);
            }

            if (setCallee(call, mlir::FlatSymbolRefAttr::get(clones[func].function)))
            {
                m_callsChanged++;
                changed = true;
            }
        }
    }

    if (!changed)
    {
        return markAllAnalysesPreserved();
    }
}
} // namespace

std::unique_ptr<mlir::Pass> pylir::Py::createMonomorphPass()
{
    return std::make_unique<Monomorph>();
}
