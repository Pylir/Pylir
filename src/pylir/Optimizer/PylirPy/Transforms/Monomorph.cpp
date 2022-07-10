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
#include <pylir/Optimizer/PylirPy/IR/TypeRefineableInterface.hpp>
#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Variant.hpp>

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

using TypeFlowArgValue = llvm::PointerUnion<mlir::SymbolRefAttr, pylir::Py::ObjectTypeInterface>;

struct FunctionSpecialization
{
    mlir::FunctionOpInterface function;
    std::vector<TypeFlowArgValue> argTypes;

    FunctionSpecialization(mlir::FunctionOpInterface function, std::vector<TypeFlowArgValue> argTypes)
        : function(function), argTypes(std::move(argTypes))
    {
    }

    bool operator==(const FunctionSpecialization& rhs) const
    {
        return std::tie(function, argTypes) == std::tie(rhs.function, rhs.argTypes);
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
        return {llvm::DenseMapInfo<mlir::FunctionOpInterface>::getEmptyKey(), {}};
    }

    static inline FunctionSpecialization getTombstoneKey()
    {
        return {llvm::DenseMapInfo<mlir::FunctionOpInterface>::getTombstoneKey(), {}};
    }

    static inline unsigned getHashValue(const FunctionSpecialization& value)
    {
        auto f = [](TypeFlowArgValue unionVal) { return (std::uintptr_t)unionVal.getOpaqueValue(); };
        auto argTypes = value.argTypes;
        return llvm::hash_combine(&*value.function, llvm::hash_combine_range(llvm::mapped_iterator(argTypes.begin(), f),
                                                                             llvm::mapped_iterator(argTypes.end(), f)));
    }

    static inline bool isEqual(const FunctionSpecialization& lhs, const FunctionSpecialization& rhs)
    {
        return lhs == rhs;
    }
};

template <>
struct llvm::DenseMapInfo<std::vector<pylir::Py::TypeAttrUnion>>
{
    static inline std::vector<pylir::Py::TypeAttrUnion> getEmptyKey()
    {
        return {llvm::DenseMapInfo<pylir::Py::TypeAttrUnion>::getEmptyKey()};
    }

    static inline std::vector<pylir::Py::TypeAttrUnion> getTombstoneKey()
    {
        return {llvm::DenseMapInfo<pylir::Py::TypeAttrUnion>::getTombstoneKey()};
    }

    static inline unsigned getHashValue(const std::vector<pylir::Py::TypeAttrUnion>& value)
    {
        return llvm::hash_combine_range(value.begin(), value.end());
    }

    static inline bool isEqual(const std::vector<pylir::Py::TypeAttrUnion>& lhs,
                               const std::vector<pylir::Py::TypeAttrUnion>& rhs)
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
    llvm::SmallVector<mlir::Block*, 2> successors{};
    mlir::Block* skippedBlock = nullptr;
};

using ExecutionResult = std::variant<SuccessorBlocks, FunctionCall, llvm::SmallVector<pylir::Py::ObjectTypeInterface>>;

class ExecutionFrame
{
    /// Op that is about to be executed.
    mlir::Operation* m_nextExecutedOp;
    /// Mapping of TypeFlowIR meta values and their actual current values.
    llvm::DenseMap<mlir::Value, pylir::Py::TypeAttrUnion> m_values;

public:
    explicit ExecutionFrame(mlir::Operation* nextExecutedOp,
                            llvm::DenseMap<mlir::Value, pylir::Py::TypeAttrUnion> valuesInit = decltype(m_values)(0))
        : m_nextExecutedOp(nextExecutedOp), m_values(std::move(valuesInit))
    {
    }

    [[nodiscard]] mlir::Operation& getNextExecutedOp() const
    {
        return *m_nextExecutedOp;
    }

    [[nodiscard]] llvm::DenseMap<mlir::Value, pylir::Py::TypeAttrUnion>& getValues()
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
    ExecutionResult execute(mlir::Operation* context, mlir::SymbolTableCollection& collection)
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
                            llvm::SmallVector<pylir::TypeFlow::OpFoldResult> result;
                            if (mlir::failed(interface.exec(
                                    llvm::to_vector(llvm::map_range(interface->getOperands(),
                                                                    [&](mlir::Value val) { return m_values[val]; })),
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
                                    if (auto attr = value.dyn_cast<pylir::Py::TypeAttrUnion>())
                                    {
                                        m_values[res] = attr;
                                    }
                                    else if (auto val = value.dyn_cast<mlir::Value>())
                                    {
                                        m_values[res] = m_values[val];
                                    }
                                    else
                                    {
                                        m_values[res] = {};
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
                                    {boolean.getValue() ? condBranchOp.getTrueSucc() : condBranchOp.getFalseSucc()},
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
                                auto& lookup = m_values[value];
                                if (auto type = lookup.dyn_cast_or_null<pylir::Py::ObjectTypeInterface>())
                                {
                                    returnValue = type;
                                }
                                else if (lookup.isa<pylir::Py::ObjectAttrInterface, mlir::SymbolRefAttr,
                                                    pylir::Py::UnboundAttr>())
                                {
                                    returnValue =
                                        pylir::Py::typeOfConstant(lookup.cast<mlir::Attribute>(), collection, context);
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
                                    context, callOp.getCallee());
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
                                            collection.lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(context,
                                                                                                         callee);
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
                                    context, funcAttr.getValue());
                            }

                            std::vector<TypeFlowArgValue> arguments(callOp.getArguments().size());
                            for (auto [argValue, inValue] : llvm::zip(arguments, callOp.getArguments()))
                            {
                                auto value = m_values[inValue];
                                // We only allow references referring to type object to be passed as objects across
                                // function boundaries. Everything else has to be a type.
                                if (auto ref = value.template dyn_cast_or_null<mlir::SymbolRefAttr>())
                                {
                                    auto lookup =
                                        collection.lookupNearestSymbolFrom<pylir::Py::GlobalValueOp>(context, ref);
                                    if (lookup
                                        && lookup.getInitializerAttr().template isa_and_nonnull<pylir::Py::TypeAttr>())
                                    {
                                        argValue = ref;
                                        continue;
                                    }
                                }

                                if (auto type = value.template dyn_cast_or_null<pylir::Py::ObjectTypeInterface>())
                                {
                                    argValue = type;
                                }
                                else if (value.template isa_and_nonnull<pylir::Py::ObjectAttrInterface,
                                                                        mlir::SymbolRefAttr, pylir::Py::UnboundAttr>())
                                {
                                    argValue = pylir::Py::typeOfConstant(value.template cast<mlir::Attribute>(),
                                                                         collection, context);
                                }
                            }
                            return FunctionCall{FunctionSpecialization{function, std::move(arguments)},
                                                callOp.getResults(), callOp.getInstruction()};
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

class Orchestrator;

struct CallWaiting
{
    ExecutionFrame frame;
    Orchestrator* orchestrator;
    mlir::ValueRange resultValues;

    CallWaiting(ExecutionFrame&& frame, Orchestrator* orchestrator, mlir::ValueRange resultValues)
        : frame(std::move(frame)), orchestrator(orchestrator), resultValues(resultValues)
    {
    }
};

struct Loop
{
    llvm::SetVector<mlir::Block*> exitBlocks;
    llvm::DenseSet<std::vector<pylir::Py::TypeAttrUnion>> headerArgs;

    Loop() : exitBlocks({}) {}
};

struct RecursionInfo
{
    enum class Phase
    {
        Expansion,
        Breaking
    };

    struct SavePoint
    {
        CallWaiting callWaiting;
        std::vector<Loop> loopState;
    };

    Phase phase = Phase::Expansion;
    llvm::DenseMap<Orchestrator*, std::vector<SavePoint>> containedOrchsToCycleCalls;
    Orchestrator* broken = nullptr;
    llvm::DenseSet<std::vector<pylir::Py::TypeAttrUnion>> seen;

    explicit RecursionInfo(llvm::DenseMap<Orchestrator*, std::vector<SavePoint>>&& containedOrchsToCycleCalls)
        : containedOrchsToCycleCalls(std::move(containedOrchsToCycleCalls))
    {
    }
};

/// Handles scheduling on the basic block level of a specific function call. It executes ExecutionFrames, which usually
/// execute a basic block, and then handles scheduling which basic blocks should be executed next.
class Orchestrator
{
    mlir::Operation* m_context;
    pylir::Py::TypeFlow& m_typeFlowIR;
    pylir::LoopInfo& m_loopInfo;
    mlir::DominanceInfo& m_dominanceInfo;
    mlir::Liveness& m_liveness;

    std::vector<pylir::Py::ObjectTypeInterface> m_returnTypes;
    llvm::DenseMap<mlir::Block*, bool> m_finishedBlocks;
    llvm::DenseMap<mlir::Value, pylir::Py::TypeAttrUnion> m_values;

    llvm::DenseMap<pylir::Loop*, Loop> m_loops;
    llvm::DenseMap<mlir::Operation*, FunctionSpecialization> m_callSites;

    std::size_t m_inQueueCount = 0;
    std::vector<CallWaiting> m_waitingCallers;
    llvm::SetVector<Orchestrator*> m_activeCalls;

    std::vector<std::shared_ptr<RecursionInfo>> m_recursionInfos;

    friend struct llvm::GraphTraits<Orchestrator*>;
    friend struct llvm::GraphTraits<llvm::Inverse<Orchestrator*>>;

    /// Creates successor frames for the given blocks. If the block is a loop header, its loop is not NULL and also
    /// passed.
    std::vector<ExecutionFrame> buildSuccessorFrames(llvm::ArrayRef<std::pair<mlir::Block*, pylir::Loop*>> blocks,
                                                     mlir::SymbolTableCollection& collection)
    {
        std::vector<ExecutionFrame> results;
        for (auto [iter, loop] : blocks)
        {
            std::vector<pylir::Py::TypeAttrUnion> blockArgs(iter->getNumArguments());
            llvm::DenseMap<mlir::Value, pylir::Py::TypeAttrUnion> values;
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
                        existing->second = existing->second.join(succValue, collection, m_context);
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
            auto temp = handleNoAndIntoLoopSuccessors(loopOrch.exitBlocks.getArrayRef(), collection);
            results.insert(results.end(), std::move_iterator(temp.begin()), std::move_iterator(temp.end()));
            m_loops.erase(loop);
        }
        return results;
    }

    std::vector<ExecutionFrame> handleNoAndIntoLoopSuccessors(mlir::BlockRange successors,
                                                              mlir::SymbolTableCollection& collection)
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
        return buildSuccessorFrames(successor, collection);
    }

    friend class InQueueCount;

public:
    [[nodiscard]] bool finishedExecution() const
    {
        return m_inQueueCount == 0 && m_activeCalls.empty();
    }

    [[nodiscard]] bool inCalls() const
    {
        return !m_activeCalls.empty();
    }

    [[nodiscard]] bool inQueue() const
    {
        return m_inQueueCount;
    }

    void addWaitingCall(ExecutionFrame&& frame, Orchestrator* orch, mlir::ValueRange results)
    {
        m_waitingCallers.emplace_back(std::move(frame), orch, results);
        orch->m_activeCalls.insert(this);
    }

    void addWaitingCall(const CallWaiting& callWaiting)
    {
        m_waitingCallers.push_back(callWaiting);
        callWaiting.orchestrator->m_activeCalls.insert(this);
    }

    std::vector<CallWaiting>& getWaitingCallers()
    {
        return m_waitingCallers;
    }

    [[nodiscard]] const auto& getActiveCalls() const
    {
        return m_activeCalls;
    }

    [[nodiscard]] bool inRecursions() const
    {
        return !m_recursionInfos.empty();
    }

    void addRecursionInfo(std::shared_ptr<RecursionInfo> info)
    {
        m_recursionInfos.push_back(std::move(info));
    }

    llvm::ArrayRef<std::shared_ptr<RecursionInfo>> getRecursionInfos()
    {
        return m_recursionInfos;
    }

    llvm::ArrayRef<pylir::Py::ObjectTypeInterface> getReturnTypes() const
    {
        return m_returnTypes;
    }

    [[nodiscard]] bool hasPreliminaryReturnTypes() const
    {
        return !m_recursionInfos.empty() && m_recursionInfos.back()->broken == this;
    }

    [[nodiscard]] const llvm::DenseMap<mlir::Value, pylir::Py::TypeAttrUnion>& getValues() const
    {
        return m_values;
    }

    [[nodiscard]] const llvm::DenseMap<mlir::Operation*, FunctionSpecialization>& getCallSites() const
    {
        return m_callSites;
    }

    explicit Orchestrator(mlir::Operation* context, TypeFlowInstance& instance)
        : m_context(context),
          m_typeFlowIR(instance.typeFLowIR),
          m_loopInfo(instance.loopInfo),
          m_dominanceInfo(instance.dominanceInfo),
          m_liveness(instance.liveness),
          m_returnTypes(m_typeFlowIR.getFunction().getNumResults(),
                        pylir::Py::UnboundType::get(m_typeFlowIR.getFunction()->getContext()))
    {
    }

    [[nodiscard]] mlir::Block* getEntryBlock() const
    {
        return &m_typeFlowIR.getFunction().front();
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
        auto result = executionFrame.execute(m_context, symbolTableCollection);
        if (auto* call = std::get_if<FunctionCall>(&result))
        {
            auto [existing, inserted] = m_callSites.insert({call->callOp, call->functionSpecialization});
            if (!inserted && existing->second != call->functionSpecialization)
            {
                existing->second = FunctionSpecialization{nullptr, {}};
            }
            return {std::move(*call)};
        }

        // Save this blocks result first of all.
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
        if (successorBlocks.skippedBlock)
        {
            llvm::SmallPtrSet<mlir::Block*, 2> set(successorBlocks.successors.begin(),
                                                   successorBlocks.successors.end());
            frames = skipDominating(successorBlocks.skippedBlock, symbolTableCollection, set);
        }

        auto* loop = m_loopInfo.getLoopFor(block);
        if (!loop)
        {
            auto temp = handleNoAndIntoLoopSuccessors(successorBlocks.successors, symbolTableCollection);
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

        auto temp = buildSuccessorFrames(successors, symbolTableCollection);
        frames.insert(frames.end(), std::move_iterator(temp.begin()), std::move_iterator(temp.end()));
        return frames;
    }

    std::vector<ExecutionFrame> skipDominating(
        mlir::Block* root, mlir::SymbolTableCollection& collection,
        const llvm::SmallPtrSetImpl<mlir::Block*>& alreadyBeingScheduled = llvm::SmallPtrSet<mlir::Block*, 1>{})
    {
        // TODO: This is not correct, it will currently always mark `root` as finished/skipped even though it should
        //       only do that if all of its predecessors have marked it as such. More importantly, this function is
        //       currently being used naively to skip a block, when in reality it's a call edge getting skipped.
        if (root->getParent()->hasOneBlock())
        {
            m_finishedBlocks[root] = false;
            return {};
        }
        std::vector<ExecutionFrame> frames;

        std::vector<mlir::DominanceInfoNode*> skippedSubTrees = {m_dominanceInfo.getNode(root)};
        llvm::df_iterator_default_set<mlir::DominanceInfoNode*> visitedSet;
        while (!skippedSubTrees.empty())
        {
            auto* back = skippedSubTrees.back();
            skippedSubTrees.pop_back();
            for (auto* iter : llvm::depth_first_ext(back, visitedSet))
            {
                m_finishedBlocks[iter->getBlock()] = false;
                // If we reached a leaf we need to check whether it's successor have to either be skipped as well
                // or maybe even executed.
                if (!iter->isLeaf())
                {
                    continue;
                }
                llvm::SmallVector<std::pair<mlir::Block*, pylir::Loop*>> successors;
                for (auto* succ : iter->getBlock()->getSuccessors())
                {
                    if (m_finishedBlocks.count(succ) || alreadyBeingScheduled.contains(succ))
                    {
                        continue;
                    }
                    auto* succLoop = m_loopInfo.getLoopFor(succ);
                    bool skipping = true;
                    bool readyToSchedule = true;
                    for (auto* pred : succ->getPredecessors())
                    {
                        auto res = m_finishedBlocks.find(pred);
                        if (res != m_finishedBlocks.end())
                        {
                            skipping = skipping && !res->second;
                            continue;
                        }
                        // If this a loop header then a predecessor within the loop doesn't have to have finished.
                        if (succLoop && succLoop->contains(pred))
                        {
                            continue;
                        }
                        readyToSchedule = false;
                        break;
                    }
                    if (!readyToSchedule)
                    {
                        continue;
                    }
                    // If all predecessors were skipped we shall continue skipping.
                    if (skipping)
                    {
                        skippedSubTrees.push_back(m_dominanceInfo.getNode(succ));
                        continue;
                    }
                    // Otherwise all predecessors are either ready or skipped and we can schedule.
                    successors.emplace_back(succ, succLoop);
                }
                auto temp = buildSuccessorFrames(successors, collection);
                frames.insert(frames.end(), std::move_iterator(temp.begin()), std::move_iterator(temp.end()));
            }
        }
        return frames;
    }

    void removeActiveCaller(Orchestrator* orch)
    {
        m_activeCalls.remove(orch);
    }

    void retire()
    {
        for (auto& iter : m_waitingCallers)
        {
            iter.orchestrator->removeActiveCaller(this);
        }
        m_waitingCallers.clear();
        m_waitingCallers.shrink_to_fit();
        m_finishedBlocks.shrink_and_clear();
        m_loops.shrink_and_clear();
        m_activeCalls.clear();
        m_recursionInfos.clear();
        m_recursionInfos.shrink_to_fit();
    }

    std::vector<Loop> getLoopState(mlir::Block* block)
    {
        std::vector<Loop> state;
        for (auto* loop = m_loopInfo.getLoopFor(block); loop; loop = loop->getParentLoop())
        {
            auto res = m_loops.find(loop);
            if (res == m_loops.end())
            {
                break;
            }
            state.push_back(res->second);
        }
        return state;
    }

    void installLoopState(mlir::Block* block, const std::vector<Loop>& vec)
    {
        std::size_t index = 0;
        for (auto* loop = m_loopInfo.getLoopFor(block); loop; loop = loop->getParentLoop())
        {
            PYLIR_ASSERT(index < vec.size());
            m_loops[loop] = vec[index++];
        }
    }
};

} // namespace

template <>
struct llvm::GraphTraits<Orchestrator*>
{
    using NodeRef = Orchestrator*;
    using ChildIteratorType = decltype(std::declval<Orchestrator&>().m_activeCalls)::iterator;

    static NodeRef getEntryNode(Orchestrator* orch)
    {
        return orch;
    }

    static ChildIteratorType child_begin(Orchestrator* orch)
    {
        return orch->m_activeCalls.begin();
    }

    static ChildIteratorType child_end(Orchestrator* orch)
    {
        return orch->m_activeCalls.end();
    }
};

template <>
struct llvm::GraphTraits<llvm::Inverse<Orchestrator*>>
{
    using NodeRef = Orchestrator*;

    static Orchestrator* map(const CallWaiting& call)
    {
        return call.orchestrator;
    }

    using ChildIteratorType = llvm::mapped_iterator<std::vector<CallWaiting>::iterator, decltype(&map)>;

    static NodeRef getEntryNode(Inverse<Orchestrator*> g)
    {
        return g.Graph;
    }

    static ChildIteratorType child_begin(Orchestrator* orch)
    {
        return {orch->m_waitingCallers.begin(), map};
    }

    static ChildIteratorType child_end(Orchestrator* orch)
    {
        return {orch->m_waitingCallers.end(), map};
    }
};

namespace
{

class InQueueCount
{
    Orchestrator* m_orchestrator = nullptr;

public:
    InQueueCount() = default;
    InQueueCount(std::nullptr_t) = delete;

    explicit InQueueCount(Orchestrator* orchestrator) : m_orchestrator(orchestrator)
    {
        m_orchestrator->m_inQueueCount++;
    }

    std::size_t release()
    {
        PYLIR_ASSERT(m_orchestrator);
        return --std::exchange(m_orchestrator, nullptr)->m_inQueueCount;
    }

    ~InQueueCount()
    {
        if (m_orchestrator)
        {
            m_orchestrator->m_inQueueCount--;
        }
    }

    InQueueCount(const InQueueCount& rhs) : m_orchestrator(rhs.m_orchestrator)
    {
        m_orchestrator->m_inQueueCount++;
    }

    InQueueCount& operator=(const InQueueCount& rhs)
    {
        if (this == &rhs)
        {
            return *this;
        }
        if (m_orchestrator)
        {
            m_orchestrator->m_inQueueCount--;
        }
        m_orchestrator = rhs.m_orchestrator;
        m_orchestrator->m_inQueueCount++;
        return *this;
    }

    InQueueCount(InQueueCount&& rhs) noexcept : m_orchestrator(std::exchange(rhs.m_orchestrator, nullptr)) {}

    InQueueCount& operator=(InQueueCount&& rhs) noexcept
    {
        if (this == &rhs)
        {
            return *this;
        }
        if (m_orchestrator)
        {
            m_orchestrator->m_inQueueCount--;
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

template <class F>
class ActionIterator
{
    F* f;

public:
    using iterator_category = std::output_iterator_tag;
    using value_type = void;
    using pointer = void;
    using reference = void;
    using difference_type = std::ptrdiff_t;

    explicit ActionIterator(F&& f) : f(&f) {}

    explicit ActionIterator(F& f) : f(&f) {}

    template <class T, std::enable_if_t<std::is_invocable_v<F, T>>* = nullptr>
    ActionIterator& operator=(T&& t)
    {
        (*f)(std::forward<T>(t));
        return *this;
    }

    ActionIterator& operator*()
    {
        return *this;
    }

    ActionIterator& operator++()
    {
        return *this;
    }

    ActionIterator operator++(int)
    {
        return *this;
    }
};

template <class NodeRef, class F>
class FilteredDFIteratorSet : public llvm::df_iterator_default_set<NodeRef>
{
    F m_f;

    using BaseSet = llvm::df_iterator_default_set<NodeRef>;
    using iterator = typename BaseSet::iterator;

public:
    explicit FilteredDFIteratorSet(F f) : m_f(std::move(f)) {}

    std::pair<iterator, bool> insert(NodeRef N)
    {
        auto [iter, inserted] = BaseSet::insert(N);
        if (!m_f(N))
        {
            return {iter, false};
        }
        return {iter, inserted};
    }

    template <typename IterT>
    void insert(IterT Begin, IterT End)
    {
        BaseSet::insert(Begin, End);
    }

    void completed(NodeRef) {}
};

template <class F>
FilteredDFIteratorSet(F) -> FilteredDFIteratorSet<typename llvm::function_traits<F>::template arg_t<0>, F>;

/// Responsible for managing Orchestrators and their execution.
class Scheduler
{
    llvm::DenseMap<mlir::FunctionOpInterface, std::unique_ptr<TypeFlowInstance>> m_typeFlowInstances;
    llvm::MapVector<FunctionSpecialization, std::unique_ptr<Orchestrator>> m_orchestrators;

    using Queue = std::queue<std::pair<ExecutionFrame, InQueueCount>>;

    std::unique_ptr<Orchestrator> createOrchestrator(mlir::FunctionOpInterface function,
                                                     mlir::AnalysisManager moduleManager)
    {
        auto [iter, inserted] = m_typeFlowInstances.insert({function, nullptr});
        if (inserted)
        {
            auto& typeFlowIR = moduleManager.getChildAnalysis<pylir::Py::TypeFlow>(function);
            mlir::ModuleAnalysisManager typeFlowModuleAnalysis(typeFlowIR.getFunction(), nullptr);
            mlir::AnalysisManager typeFlowAnalysisManager = typeFlowModuleAnalysis;
            iter->second = std::make_unique<TypeFlowInstance>(
                typeFlowIR, std::move(typeFlowAnalysisManager.getAnalysis<pylir::LoopInfo>()),
                std::move(typeFlowAnalysisManager.getAnalysis<mlir::DominanceInfo>()),
                std::move(typeFlowAnalysisManager.getAnalysis<mlir::Liveness>()));
        }
        return std::make_unique<Orchestrator>(function, *iter->second);
    }

    void handleRecursion(const llvm::SmallPtrSetImpl<Orchestrator*>& cycle, Queue& queue,
                         mlir::SymbolTableCollection& collection)
    {
        llvm::DenseMap<Orchestrator*, std::vector<RecursionInfo::SavePoint>> containedOrchs;

        for (auto* thisOrch : cycle)
        {
            for (const auto& predOrch : thisOrch->getWaitingCallers())
            {
                if (cycle.contains(predOrch.orchestrator))
                {
                    auto* block = predOrch.frame.getNextExecutedOp().getBlock();
                    containedOrchs[thisOrch].push_back({predOrch, predOrch.orchestrator->getLoopState(block)});
                    auto frames = predOrch.orchestrator->skipDominating(block, collection);
                    for (auto& frame : frames)
                    {
                        queue.emplace(std::move(frame), predOrch.orchestrator);
                    }
                    break;
                }
            }
        }
        auto info = std::make_shared<RecursionInfo>(std::move(containedOrchs));
        for (auto* orch : cycle)
        {
            orch->addRecursionInfo(info);
        }
    }

    // A dynamic topological sort algorithm for directed acyclic graphs
    // David J. Pearce, Paul H. J. Kelly
    // Journal of Experimental Algorithmics (JEA) JEA Homepage archive
    // Volume 11, 2006, Article No. 1.7
    std::vector<Orchestrator*> m_indexToOrch;
    llvm::DenseMap<Orchestrator*, std::size_t> m_orchToIndex;

    std::size_t getIndex(Orchestrator* orch)
    {
        auto [iter, inserted] = m_orchToIndex.insert({orch, 0});
        if (!inserted)
        {
            return iter->second;
        }
        m_indexToOrch.push_back(orch);
        return iter->second = m_indexToOrch.size() - 1;
    }

    llvm::SmallPtrSet<Orchestrator*, 4> addEdge(Orchestrator* from, Orchestrator* to)
    {
        auto lb = getIndex(to);
        auto ub = getIndex(from);
        if (lb >= ub)
        {
            return {};
        }

        std::vector<std::size_t> rf;
        std::vector<std::size_t> rb;
        bool forward = true;
        FilteredDFIteratorSet set([&forward, ub, lb, this](Orchestrator* orch)
                                  { return forward ? getIndex(orch) <= ub : getIndex(orch) > lb; });
        if (auto cycle = dfsForward(to, rf, ub, set); !cycle.empty())
        {
            return cycle;
        }
        forward = false;
        dfsBackward(from, rb, set);

        std::vector<Orchestrator*> reorder;

        auto pushReorder = [&](std::vector<std::size_t>& vec)
        {
            std::sort(vec.begin(), vec.end());
            std::transform(vec.begin(), vec.end(), std::back_inserter(reorder),
                           [&](std::size_t iter) { return m_indexToOrch[iter]; });
        };

        pushReorder(rb);
        pushReorder(rf);

        std::merge(rb.begin(), rb.end(), rf.begin(), rf.end(),
                   ActionIterator(
                       [&, i = 0](std::size_t index) mutable
                       {
                           m_orchToIndex[reorder[i]] = index;
                           m_indexToOrch[index] = reorder[i];
                           i++;
                       }));
        return {};
    }

    template <class F>
    llvm::SmallPtrSet<Orchestrator*, 4> dfsForward(Orchestrator* curr, std::vector<std::size_t>& r, std::size_t bound,
                                                   FilteredDFIteratorSet<Orchestrator*, F>& set)
    {
        auto range = llvm::depth_first_ext(curr, set);
        for (auto iter = range.begin(); iter != range.end(); iter++)
        {
            auto* orch = *iter;
            if (getIndex(orch) == bound)
            {
                llvm::SmallPtrSet<Orchestrator*, 4> res;
                llvm::transform(llvm::seq<std::size_t>(0, iter.getPathLength()), std::inserter(res, res.begin()),
                                [&](std::size_t index) { return iter.getPath(index); });
                return res;
            }
            r.push_back(getIndex(orch));
        }
        return {};
    }

    template <class F>
    void dfsBackward(Orchestrator* curr, std::vector<std::size_t>& r, FilteredDFIteratorSet<Orchestrator*, F>& set)
    {
        for (auto* orch : llvm::inverse_depth_first_ext(curr, set))
        {
            r.push_back(getIndex(orch));
        }
    }

    void scheduleWaitingCallsInRecursion(Queue& queue, RecursionInfo* info, Orchestrator* orch)
    {
        auto iter = llvm::remove_if(orch->getWaitingCallers(), [&](const CallWaiting& callWaiting)
                                    { return info->containedOrchsToCycleCalls.count(callWaiting.orchestrator); });
        std::for_each(iter, orch->getWaitingCallers().end(),
                      [&](CallWaiting& iter)
                      {
                          for (auto [dest, value] : llvm::zip(iter.resultValues, orch->getReturnTypes()))
                          {
                              iter.frame.getValues()[dest] = value;
                          }
                          queue.emplace(std::move(iter.frame), iter.orchestrator);
                          iter.orchestrator->removeActiveCaller(orch);
                      });
        orch->getWaitingCallers().erase(iter, orch->getWaitingCallers().end());
    }

    void checkLastInRecursion(Orchestrator* orch, Queue& queue)
    {
        for (const auto& info : llvm::reverse(orch->getRecursionInfos()))
        {
            if (info->phase != RecursionInfo::Phase::Expansion)
            {
                continue;
            }

            for (auto* iter : llvm::make_first_range(info->containedOrchsToCycleCalls))
            {
                if (iter->inQueue()
                    || llvm::any_of(iter->getActiveCalls(), [&](Orchestrator* callee)
                                    { return !info->containedOrchsToCycleCalls.count(callee); }))
                {
                    return;
                }
            }

            // We try to find the most specific return type by all orchestrators. For that purpose we rank them
            // by checking each pair of return types, how specific each is. One candidate may have less return types
            // than the other in which case the remaining ones are not considered. If the new candidate had more
            // more-specific return types than the existing best candidate, it becomes the new best candidate.
            // It doesn't actually matter which we select for correctness, just for speed of convergence.
            Orchestrator* bestCandidate = nullptr;
            for (auto& iter : llvm::make_first_range(info->containedOrchsToCycleCalls))
            {
                if (!bestCandidate)
                {
                    bestCandidate = iter;
                    continue;
                }

                std::ptrdiff_t sum = 0;
                for (auto [cand, existing] : llvm::zip(iter->getReturnTypes(), bestCandidate->getReturnTypes()))
                {
                    if (!cand)
                    {
                        if (existing)
                        {
                            sum += 1;
                        }
                        continue;
                    }
                    if (!existing)
                    {
                        sum -= 1;
                        continue;
                    }
                    if (pylir::Py::isMoreSpecific(cand, existing))
                    {
                        sum += 1;
                    }
                    else if (pylir::Py::isMoreSpecific(existing, cand))
                    {
                        sum -= 1;
                    }
                }
                if (sum <= 0)
                {
                    continue;
                }
                bestCandidate = iter;
            }
            PYLIR_ASSERT(bestCandidate);

            info->phase = RecursionInfo::Phase::Breaking;
            info->broken = bestCandidate;
            info->seen.insert({bestCandidate->getReturnTypes().begin(), bestCandidate->getReturnTypes().end()});
            scheduleWaitingCallsInRecursion(queue, info.get(), bestCandidate);
            return;
        }
    }

public:

    /// Run the typeflow analysis starting from the given root functions. These may not take any DynamicType function
    /// arguments.
    void run(llvm::ArrayRef<mlir::FunctionOpInterface> roots, mlir::AnalysisManager moduleManager)
    {
        Queue queue;

        for (auto iter : roots)
        {
            auto spec = FunctionSpecialization(iter, {});
            auto function = spec.function;
            auto& orchestrator =
                m_orchestrators.insert({spec, createOrchestrator(function, moduleManager)}).first->second;
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
                        queue.emplace(std::move(iter), front.second);
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

                if (orch->inCalls())
                {
                    // May not be queued up anymore but there are still calls we are waiting for.
                    if (orch->inRecursions())
                    {
                        checkLastInRecursion(orch, queue);
                    }
                    continue;
                }

                // Truly finished case.
                // If not part of a recursion just schedule all waiting calls.
                if (!orch->inRecursions())
                {
                    for (auto& iter : std::move(*orch).getWaitingCallers())
                    {
                        for (auto [dest, value] : llvm::zip(iter.resultValues, orch->getReturnTypes()))
                        {
                            iter.frame.getValues()[dest] = value;
                        }
                        queue.emplace(std::move(iter.frame), iter.orchestrator);
                    }
                    orch->retire();
                    continue;
                }

                // Otherwise we schedule all the other calls part of the recursive cycle until we processed the whole
                // cycle.
                const auto& info = orch->getRecursionInfos().back();
                if (info->broken != orch)
                {
                    scheduleWaitingCallsInRecursion(queue, info.get(), orch);
                    continue;
                }

                // The return type has not changed and the recursion was properly resolved. Time to retire the whole
                // recursion.
                if (!info->seen.insert({orch->getReturnTypes().begin(), orch->getReturnTypes().end()}).second)
                {
                    std::shared_ptr<RecursionInfo> infoKeepAlive = info;
                    for (auto* retiringOrch : llvm::make_first_range(infoKeepAlive->containedOrchsToCycleCalls))
                    {
                        if (retiringOrch->getRecursionInfos().back() != infoKeepAlive)
                        {
                            continue;
                        }

                        for (auto& iter : std::move(*retiringOrch).getWaitingCallers())
                        {
                            for (auto [dest, value] : llvm::zip(iter.resultValues, retiringOrch->getReturnTypes()))
                            {
                                iter.frame.getValues()[dest] = value;
                            }
                            queue.emplace(std::move(iter.frame), iter.orchestrator);
                        }
                        retiringOrch->retire();
                    }
                    continue;
                }

                for (auto& [cycleOrch, calls] : info->containedOrchsToCycleCalls)
                {
                    for (auto& call : calls)
                    {
                        cycleOrch->addWaitingCall(call.callWaiting);
                        call.callWaiting.orchestrator->installLoopState(
                            call.callWaiting.frame.getNextExecutedOp().getBlock(), call.loopState);
                    }
                }
                scheduleWaitingCallsInRecursion(queue, info.get(), info->broken);
                continue;
            }
            auto& call = pylir::get<FunctionCall>(result);
            auto [existing, inserted] = m_orchestrators.insert({std::move(call.functionSpecialization), nullptr});
            if (inserted)
            {
                existing->second = createOrchestrator(existing->first.function, moduleManager);
            }
            else if (existing->second->finishedExecution() || existing->second->hasPreliminaryReturnTypes())
            {
                // This orchestrator has already finished execution and there is no need to wait.
                for (auto [dest, value] : llvm::zip(call.resultValues, existing->second->getReturnTypes()))
                {
                    front.first.getValues()[dest] = value;
                }
                queue.emplace(std::move(front));
                continue;
            }

            auto* orch = front.second.get();
            if (!call.resultValues.empty())
            {
                existing->second->addWaitingCall(std::move(front.first), orch, call.resultValues);
                if (orch != existing->second.get())
                {
                    if (auto cycle = addEdge(orch, existing->second.get()); !cycle.empty())
                    {
                        handleRecursion(cycle, queue, collection);
                    }
                }
                else
                {
                    handleRecursion(llvm::SmallPtrSet<Orchestrator*, 1>{orch}, queue, collection);
                }
            }
            else
            {
                // No need to wait for the call if it has no results for us.
                queue.emplace(std::move(front.first), front.second);
            }

            // This is not the first call to that function, it has not yet finished execution, and we have already
            // registered ourselves as dependent. There is nothing more to do but wait for its completion.
            if (!inserted)
            {
                // If this isn't the last queue item, or we are not part of a recursion there is nothing to do but wait
                // for the calls to finish.
                if (front.second.release() != 0 || !orch->inRecursions())
                {
                    continue;
                }

                checkLastInRecursion(orch, queue);
                continue;
            }

            // First call, set up the function arguments.
            llvm::DenseMap<mlir::Value, pylir::Py::TypeAttrUnion> entryValues;
            for (auto [arg, value] :
                 llvm::zip(existing->second->getEntryBlock()->getArguments(), existing->first.argTypes))
            {
                if (auto ref = value.dyn_cast<mlir::SymbolRefAttr>())
                {
                    entryValues[arg] = ref;
                }
                else if (auto type = value.dyn_cast<pylir::Py::ObjectTypeInterface>())
                {
                    entryValues[arg] = type;
                }
            }
            queue.emplace(ExecutionFrame(&existing->second->getEntryBlock()->front(), std::move(entryValues)),
                          existing->second.get());
        }
    }

    llvm::MapVector<FunctionSpecialization, std::unique_ptr<Orchestrator>>& getResults()
    {
        return m_orchestrators;
    }
};

bool calleeDiffers(mlir::Operation* op, mlir::FlatSymbolRefAttr callee)
{
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
        .Case<pylir::Py::CallOp, pylir::Py::InvokeOp>([&](auto op) { return op.getCalleeAttr() != callee; })
        .Case<pylir::Py::FunctionCallOp, pylir::Py::FunctionInvokeOp>(
            [&](auto op)
            {
                // If it hasn't been turned into a constant than redirecting it to a clone of the given constant
                // is not valid either.
                return mlir::matchPattern(op.getFunction(), mlir::m_Constant());
            });
}

// Returns the new call op if one had to be created to replace op.
mlir::Operation* setCallee(mlir::Operation* op, mlir::FlatSymbolRefAttr callee)
{
    return llvm::TypeSwitch<mlir::Operation*, mlir::Operation*>(op)
        .Case<pylir::Py::CallOp, pylir::Py::InvokeOp>(
            [&](auto&& op)
            {
                op.setCalleeAttr(callee);
                return nullptr;
            })
        .Case(
            [&](pylir::Py::FunctionCallOp op)
            {
                mlir::OpBuilder builder(op);
                auto newCall =
                    builder.create<pylir::Py::CallOp>(op.getLoc(), op->getResultTypes(), callee, op.getCallOperands());
                newCall->setAttr(pylir::Py::alwaysBoundAttr, builder.getUnitAttr());
                op->replaceAllUsesWith(newCall);
                op->erase();
                return newCall;
            })
        .Case(
            [&](pylir::Py::FunctionInvokeOp op)
            {
                mlir::OpBuilder builder(op);
                auto newCall = builder.create<pylir::Py::InvokeOp>(
                    op.getLoc(), op->getResultTypes(), callee, op.getCallOperands(), op.getNormalDestOperands(),
                    op.getUnwindDestOperands(), op.getHappyPath(), op.getExceptionPath());
                newCall->setAttr(pylir::Py::alwaysBoundAttr, builder.getUnitAttr());
                op->replaceAllUsesWith(newCall);
                op->erase();
                return newCall;
            });
}

void Monomorph::runOnOperation()
{
    llvm::SetVector<mlir::FunctionOpInterface> roots;
    for (auto iter : getOperation().getOps<mlir::FunctionOpInterface>())
    {
        if (llvm::none_of(iter.getArgumentTypes(), std::mem_fn(&mlir::Type::isa<pylir::Py::DynamicType>)))
        {
            roots.insert(iter);
        }
    }

    llvm::MapVector<FunctionSpecialization, std::unique_ptr<Orchestrator>> results;
    {
        Scheduler scheduler;
        scheduler.run(roots.getArrayRef(), getAnalysisManager());
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
        clone.function = func.function;
        bool isRoot = roots.contains(func.function);
        for (const auto& [key, value] : orchestrator->getValues())
        {
            auto attr = value.dyn_cast_or_null<mlir::Attribute>();
            if (!attr)
            {
                continue;
            }

            mlir::Value instrValue;
            if (auto blockArg = key.dyn_cast<mlir::BlockArgument>())
            {
                if (!blockArg.getOwner()->isEntryBlock())
                {
                    continue;
                }
                auto dynamicArgIndex = blockArg.getArgNumber();
                auto filter = llvm::make_filter_range(func.function.getArguments(), [](mlir::Value val)
                                                      { return val.getType().isa<pylir::Py::DynamicType>(); });
                instrValue = *std::next(filter.begin(), dynamicArgIndex);
            }
            else
            {
                auto mapping = key.getDefiningOp<pylir::TypeFlow::TypeFlowValueMappingInterface>();
                if (!mapping)
                {
                    continue;
                }
                instrValue = mapping.mapValue(key);
            }

            // Cloning of a function body is done lazily for the case where no value has changed.
            // Roots are not cloned but updated in place.
            if (clone.function == func.function && !isRoot)
            {
                clone.function = clone.function->clone(clone.mapping);
                mlir::cast<mlir::SymbolOpInterface>(*clone.function).setPrivate();
                table.insert(clone.function);
                m_functionsCloned++;
            }

            auto cloneValue = clone.mapping.lookupOrDefault(instrValue);
            mlir::OpBuilder builder = [=]
            {
                if (auto* op = cloneValue.getDefiningOp())
                {
                    return mlir::OpBuilder(op);
                }

                return mlir::OpBuilder::atBlockBegin(cloneValue.cast<mlir::BlockArgument>().getParentBlock());
            }();
            mlir::Dialect* dialect;
            if (cloneValue.isa<mlir::BlockArgument>())
            {
                // Due to the lack of a better way in the case of a block argument we just use PylirPyDialect for now.
                dialect = getContext().getLoadedDialect<pylir::Py::PylirPyDialect>();
            }
            else
            {
                dialect = cloneValue.getDefiningOp()->getDialect();
            }
            auto* constant = dialect->materializeConstant(builder, attr, cloneValue.getType(), cloneValue.getLoc());
            PYLIR_ASSERT(constant);
            cloneValue.replaceAllUsesWith(constant->getResult(0));
            m_valuesReplaced++;
            changed = true;
        }
    }

    // Redirect all call-sites to the specializations. This is not done naively but only if the target function is a
    // result of cloning. That is because it has only been cloned if it itself had to have changed. Since cloning itself
    // can also be caused by redirecting a call-site to a clone, we iterate through this until a fixpoint is reached.
    // This implementation could be improved via a topological sort or similar, but this will do for now.
    bool cloneOccurred;
    do
    {
        cloneOccurred = false;
        for (auto& [thisFunc, orchestrator] : results)
        {
            bool isRoot = roots.contains(thisFunc.function);
            auto& thisClone = clones[thisFunc];
            for (const auto& [origCall, func] : orchestrator->getCallSites())
            {
                auto calcCall = [&, &origCall = origCall, &thisFunc = thisFunc]
                {
                    if (thisClone.function == thisFunc.function)
                    {
                        return origCall;
                    }
                    auto& mapping = thisClone.mapping;
                    // TODO: Map call properly (not like the following code) as soon as it is supported by
                    // BlockAndValueMapping
                    if (origCall->getNumResults() != 0)
                    {
                        return mapping.lookupOrDefault(origCall->getResult(0)).getDefiningOp();
                    }

                    auto* mappedBlock = mapping.lookupOrDefault(origCall->getBlock());
                    auto distance = std::distance(origCall->getBlock()->begin(), mlir::Block::iterator{origCall});
                    return &*std::next(mappedBlock->begin(), distance);
                };

                auto& calleeClone = clones[func];
                auto* call = calcCall();
                // We only change the call-site if the callee had to be cloned, since cloned implies IR was changed
                // based on prior analysis.
                if (!func.function || calleeClone.function == func.function
                    || !calleeDiffers(call, mlir::FlatSymbolRefAttr::get(calleeClone.function)))
                {
                    continue;
                }

                // Lazy cloning of this function if it has not yet occurred. Just like in the constant setting loop.
                if (thisClone.function == thisFunc.function && !isRoot)
                {
                    thisClone.function = thisClone.function->clone(thisClone.mapping);
                    mlir::cast<mlir::SymbolOpInterface>(*thisClone.function).setPrivate();
                    table.insert(thisClone.function);
                    m_functionsCloned++;
                    cloneOccurred = true;
                    call = calcCall();
                }

                auto* newCall = setCallee(call, mlir::FlatSymbolRefAttr::get(calleeClone.function));
                if (newCall)
                {
                    // call is now invalid, but it's still contained within the mapping. Have to update it.
                    thisClone.mapping.map(origCall->getResults(), newCall->getResults());
                }
                m_callsChanged++;
                changed = true;
            }
        }
    } while (cloneOccurred);

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
