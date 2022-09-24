//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/IR/TypeRefineableInterface.hpp>
#include <pylir/Support/Functional.hpp>
#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Variant.hpp>

#include <queue>
#include <unordered_set>
#include <utility>
#include <variant>

#include "Passes.hpp"

namespace pylir::Py
{
#define GEN_PASS_DEF_MONOMORPHPASS
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"
} // namespace pylir::Py

#define DEBUG_TYPE "monomorph"

namespace
{
class Monomorph : public pylir::Py::impl::MonomorphPassBase<Monomorph>
{
protected:
    void runOnOperation() override;

public:
    using Base::Base;
};

using TypeFlowArgValue = llvm::PointerUnion<pylir::Py::RefAttr, pylir::Py::ObjectTypeInterface>;

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

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const FunctionSpecialization& specialization)
{
    os << const_cast<mlir::FunctionOpInterface&>(specialization.function).getName() << '(';
    llvm::interleaveComma(specialization.argTypes, os,
                          [&](TypeFlowArgValue value)
                          {
                              if (auto ref = value.dyn_cast<pylir::Py::RefAttr>())
                              {
                                  os << ref;
                              }
                              else if (auto type = value.dyn_cast<pylir::Py::ObjectTypeInterface>())
                              {
                                  os << type;
                              }
                              else
                              {
                                  os << "unknown";
                              }
                          });
    os << ')';
    return os;
}

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
                                    result)))
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
                                else if (lookup.isa<pylir::Py::ObjectAttrInterface, pylir::Py::RefAttr,
                                                    pylir::Py::UnboundAttr>())
                                {
                                    returnValue = pylir::Py::typeOfConstant(lookup.cast<mlir::Attribute>());
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
                                    if (auto callee = calleeValue.template dyn_cast_or_null<pylir::Py::RefAttr>())
                                    {
                                        funcAttr = callee.getSymbol()
                                                       .getInitializerAttr()
                                                       .template dyn_cast_or_null<pylir::Py::FunctionAttr>();
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
                                if (auto ref = value.template dyn_cast_or_null<pylir::Py::RefAttr>())
                                {
                                    if (ref.getSymbol()
                                            .getInitializerAttr()
                                            .template isa_and_nonnull<pylir::Py::TypeAttr>())
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
                                                                        pylir::Py::RefAttr, pylir::Py::UnboundAttr>())
                                {
                                    argValue = pylir::Py::typeOfConstant(value.template cast<mlir::Attribute>());
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
    mlir::Liveness liveness;

    TypeFlowInstance(pylir::Py::TypeFlow& typeFLowIr, pylir::LoopInfo&& loopInfo, mlir::Liveness&& liveness)
        : typeFLowIR(typeFLowIr), loopInfo(std::move(loopInfo)), liveness(std::move(liveness))
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

    bool operator==(const CallWaiting& rhs) const
    {
        return orchestrator == rhs.orchestrator && &frame.getNextExecutedOp() == &rhs.frame.getNextExecutedOp();
    }
};

struct Loop
{
    llvm::SetVector<std::pair<mlir::Block*, mlir::Block*>> exitEdges;
    llvm::DenseSet<std::vector<pylir::Py::TypeAttrUnion>> headerArgs;

    Loop() : exitEdges({}) {}
};

class OrchestratorStatepoint;

struct RecursionInfo;

/// Handles scheduling on the basic block level of a specific function call. It executes ExecutionFrames, which usually
/// execute a basic block, and then handles scheduling which basic blocks should be executed next.
class Orchestrator
{
    std::size_t m_inQueueCount = 0;
    mlir::FunctionOpInterface m_context;
    pylir::Py::TypeFlow& m_typeFlowIR;
    pylir::LoopInfo& m_loopInfo;
    mlir::Liveness& m_liveness;

    std::vector<pylir::Py::ObjectTypeInterface> m_returnTypes;
    llvm::DenseMap<std::pair<mlir::Block*, mlir::Block*>, bool> m_finishedEdges;
    llvm::DenseMap<mlir::Value, pylir::Py::TypeAttrUnion> m_values;
    llvm::DenseMap<pylir::Loop*, Loop> m_loops;
    llvm::DenseMap<mlir::Operation*, FunctionSpecialization> m_callSites;
    std::vector<CallWaiting> m_waitingCallers;
    llvm::DenseSet<Orchestrator*> m_activeCalls;

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
                if (!m_finishedEdges.lookup({*pred, iter}))
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
                // New loop iteration. Remove all outgoing edges from loop blocks to make them reevaluated properly.
                for (auto* block : loop->getBlocks())
                {
                    for (auto* succ : block->getSuccessors())
                    {
                        m_finishedEdges.erase({block, succ});
                    }
                }
                results.emplace_back(&iter->front(), std::move(values));
                continue;
            }
            // We have previously encountered this loop with these block args and hence reached a fixpoint.
            // Unleash the exits!
            for (const auto& pair : loopOrch.exitEdges)
            {
                m_finishedEdges.insert({pair, true});
            }
            auto temp = findSuccessors(llvm::make_second_range(loopOrch.exitEdges), collection);
            results.insert(results.end(), std::move_iterator(temp.begin()), std::move_iterator(temp.end()));
            m_loops.erase(loop);
        }
        return results;
    }

    template <class Range>
    std::vector<ExecutionFrame> findSuccessors(Range&& successors, mlir::SymbolTableCollection& collection)
    {
        llvm::SmallVector<std::pair<mlir::Block*, pylir::Loop*>> successor;
        for (mlir::Block* iter : successors)
        {
            auto* succLoop = m_loopInfo.getLoopFor(iter);
            if (!succLoop || succLoop->getHeader() != iter)
            {
                if (llvm::all_of(iter->getPredecessors(),
                                 [=](mlir::Block* pred) {
                                     return m_finishedEdges.count({pred, iter});
                                 }))
                {
                    successor.emplace_back(iter, nullptr);
                }
                continue;
            }

            // Loop headers are treated specially. To be able to enter the loop header for the very first iteration we
            // enter it if all entry edges are ready but exactly no back edges are. As soon as at least one backedge
            // is ready however, that means it's not the loops first iteration and therefore ALL incoming edges have
            // to be ready.
            bool anyBackEdgeReady = false;
            bool allReady = true;
            bool allEntriesReady = true;
            for (auto* pred : iter->getPredecessors())
            {
                bool edgeReady = m_finishedEdges.count({pred, iter});
                allReady = allReady && edgeReady;
                if (succLoop->contains(pred))
                {
                    anyBackEdgeReady = anyBackEdgeReady || edgeReady;
                }
                else
                {
                    allEntriesReady = allEntriesReady && edgeReady;
                }
            }
            if (allReady || (!anyBackEdgeReady && allEntriesReady))
            {
                successor.emplace_back(iter, succLoop);
            }
        }
        return buildSuccessorFrames(successor, collection);
    }

    std::vector<ExecutionFrame> skipDominating(std::pair<mlir::Block*, mlir::Block*> edge,
                                               mlir::SymbolTableCollection& collection)
    {
        LLVM_DEBUG({
            llvm::dbgs() << "Marking edge ";
            edge.first->printAsOperand(llvm::dbgs());
            llvm::dbgs() << " -> ";
            edge.second->printAsOperand(llvm::dbgs());
            llvm::dbgs() << " as skipped\n";
        });
        m_finishedEdges[edge] = false;

        std::vector<ExecutionFrame> frames;
        std::queue<mlir::Block*> skippedWorklist;

        auto add = [&](mlir::Block* toBlock)
        {
            bool completelySkipped = true;
            for (auto* pred : toBlock->getPredecessors())
            {
                auto res = m_finishedEdges.find({pred, toBlock});
                if (res == m_finishedEdges.end())
                {
                    return;
                }
                completelySkipped = completelySkipped && !res->second;
            }
            if (completelySkipped)
            {
                skippedWorklist.push(toBlock);
                return;
            }
            auto temp = findSuccessors(llvm::ArrayRef{toBlock}, collection);
            frames.insert(frames.end(), std::move_iterator(temp.begin()), std::move_iterator(temp.end()));
        };

        add(edge.second);
        while (!skippedWorklist.empty())
        {
            auto* back = skippedWorklist.back();
            skippedWorklist.pop();
            for (auto* succ : back->getSuccessors())
            {
                m_finishedEdges[{back, succ}] = false;
                LLVM_DEBUG({
                    llvm::dbgs() << "Marking edge ";
                    back->printAsOperand(llvm::dbgs());
                    llvm::dbgs() << " -> ";
                    succ->printAsOperand(llvm::dbgs());
                    llvm::dbgs() << " as skipped\n";
                });
                add(succ);
            }
        }
        return frames;
    }

    friend class InQueueCount;

public:

    [[nodiscard]] llvm::StringRef getName() const
    {
        return m_typeFlowIR.getFunction().getName();
    }

    [[nodiscard]] FunctionSpecialization getFunctionSpecialization() const
    {
        auto range = llvm::map_range(getEntryBlock()->getArguments(),
                                     [&](mlir::Value val) -> TypeFlowArgValue
                                     {
                                         auto value = getValues().lookup(val);
                                         if (!value)
                                         {
                                             return nullptr;
                                         }
                                         if (auto ref = value.dyn_cast<pylir::Py::RefAttr>())
                                         {
                                             return ref;
                                         }
                                         return value.cast<pylir::Py::ObjectTypeInterface>();
                                     });
        return FunctionSpecialization(m_context, {range.begin(), range.end()});
    }

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

    llvm::MutableArrayRef<CallWaiting> getWaitingCallers()
    {
        return m_waitingCallers;
    }

    void eraseWaitingCallers(llvm::MutableArrayRef<CallWaiting>::const_iterator begin,
                             llvm::MutableArrayRef<CallWaiting>::const_iterator end)
    {
        for (const auto* iter = begin; iter != end; iter++)
        {
            iter->orchestrator->m_activeCalls.erase(this);
        }
        m_waitingCallers.erase(m_waitingCallers.begin() + (begin - getWaitingCallers().begin()),
                               m_waitingCallers.begin() + (end - getWaitingCallers().begin()));
    }

    [[nodiscard]] const auto& getActiveCalls() const
    {
        return m_activeCalls;
    }

    llvm::MutableArrayRef<pylir::Py::ObjectTypeInterface> getReturnTypes()
    {
        return m_returnTypes;
    }

    [[nodiscard]] const llvm::DenseMap<mlir::Value, pylir::Py::TypeAttrUnion>& getValues() const
    {
        return m_values;
    }

    [[nodiscard]] const llvm::DenseMap<mlir::Operation*, FunctionSpecialization>& getCallSites() const
    {
        return m_callSites;
    }

    explicit Orchestrator(mlir::FunctionOpInterface context, TypeFlowInstance& instance)
        : m_context(context),
          m_typeFlowIR(instance.typeFLowIR),
          m_loopInfo(instance.loopInfo),
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
        {
            auto& newValues = executionFrame.getValues();
            for (auto& pair : newValues)
            {
                auto [existing, inserted] = m_values.insert(pair);
                if (!inserted)
                {
                    existing->second = pair.second;
                }
            }
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
        LLVM_DEBUG({
            if (!successorBlocks.successors.empty())
            {
                llvm::dbgs() << "Candidate successors: ";
                llvm::interleaveComma(successorBlocks.successors, llvm::dbgs(),
                                      [](mlir::Block* succ) { succ->printAsOperand(llvm::dbgs()); });
                llvm::dbgs() << '\n';
            }
        });
        if (successorBlocks.skippedBlock)
        {
            frames = skipDominating({block, successorBlocks.skippedBlock}, symbolTableCollection);
        }

        auto* loop = m_loopInfo.getLoopFor(block);
        if (!loop)
        {
            for (auto* succ : successorBlocks.successors)
            {
                m_finishedEdges[{block, succ}] = true;
            }
        }
        else
        {
            auto& loopOrch = m_loops[loop];
            llvm::SmallVector<std::pair<mlir::Block*, pylir::Loop*>> successors;
            for (auto* succ : successorBlocks.successors)
            {
                if (loop->contains(succ))
                {
                    m_finishedEdges[{block, succ}] = true;
                    continue;
                }
                loopOrch.exitEdges.insert({block, succ});
            }
        }

        auto temp = findSuccessors(successorBlocks.successors, symbolTableCollection);
        frames.insert(frames.end(), std::move_iterator(temp.begin()), std::move_iterator(temp.end()));
        return frames;
    }

    void retire()
    {
        for (auto& iter : m_waitingCallers)
        {
            iter.orchestrator->m_activeCalls.erase(this);
        }
        m_waitingCallers.clear();
        m_waitingCallers.shrink_to_fit();
        m_finishedEdges.shrink_and_clear();
        m_loops.shrink_and_clear();
        m_activeCalls.clear();
    }
};

[[maybe_unused]] llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Orchestrator& orchestrator)
{
    if (orchestrator.getValues().empty())
    {
        return os << orchestrator.getName();
    }
    return os << orchestrator.getFunctionSpecialization();
}

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
                std::move(typeFlowAnalysisManager.getAnalysis<mlir::Liveness>()));
        }
        return std::make_unique<Orchestrator>(function, *iter->second);
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

    llvm::SmallVector<Orchestrator*> addEdge(Orchestrator* from, Orchestrator* to)
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
    llvm::SmallVector<Orchestrator*> dfsForward(Orchestrator* curr, std::vector<std::size_t>& r, std::size_t bound,
                                                FilteredDFIteratorSet<Orchestrator*, F>& set)
    {
        auto range = llvm::depth_first_ext(curr, set);
        for (auto iter = range.begin(); iter != range.end(); iter++)
        {
            auto* orch = *iter;
            if (getIndex(orch) == bound)
            {
                llvm::SmallVector<Orchestrator*> res(iter.getPathLength());
                llvm::transform(llvm::seq<std::size_t>(0, iter.getPathLength()), res.begin(),
                                pylir::bind_front(&decltype(iter)::getPath, &iter));
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

    static void scheduleWaitingCalls(Queue& queue, Orchestrator* calleeOrch)
    {
        for (auto& iter : calleeOrch->getWaitingCallers())
        {
            for (auto [dest, value] : llvm::zip(iter.resultValues, calleeOrch->getReturnTypes()))
            {
                iter.frame.getValues()[dest] = value;
            }
            LLVM_DEBUG({
                llvm::dbgs() << "Queued " << *iter.orchestrator << ": ";
                iter.frame.getNextExecutedOp().getBlock()->printAsOperand(llvm::dbgs());
                llvm::dbgs() << '\n';
            });
            queue.emplace(std::move(iter.frame), iter.orchestrator);
        }
        calleeOrch->eraseWaitingCallers(calleeOrch->getWaitingCallers().begin(), calleeOrch->getWaitingCallers().end());
    }

    std::size_t m_maxTypeSize;
    mlir::Pass::Statistic& m_typesOverDefined;

    [[nodiscard]] bool typeIsOverDefined(mlir::Type type) const
    {
        auto subElements = type.dyn_cast<mlir::SubElementTypeInterface>();
        if (!subElements)
        {
            return false;
        }
        std::size_t counter = 0;
        subElements.walkSubTypes([&](auto&&) { counter++; });
        return counter > m_maxTypeSize;
    }

public:
    Scheduler(std::size_t maxTypeSize, mlir::Pass::Statistic& typesOverDefined)
        : m_maxTypeSize(maxTypeSize), m_typesOverDefined(typesOverDefined)
    {
    }

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
            LLVM_DEBUG({
                llvm::dbgs() << "Executing " << *front.second << ": ";
                front.first.getNextExecutedOp().getBlock()->printAsOperand(llvm::dbgs());
                llvm::dbgs() << "\n";
            });
            auto result = front.second->execute(front.first, collection);
            if (auto* vec = std::get_if<std::vector<ExecutionFrame>>(&result))
            {
                if (!vec->empty())
                {
                    LLVM_DEBUG({ llvm::dbgs() << "Ready successors: \n"; });
                    // Successor blocks and this execution round was definitely not the last one from the orchestrator.
                    for (auto& iter : *vec)
                    {
                        LLVM_DEBUG({
                            llvm::dbgs() << "- ";
                            iter.getNextExecutedOp().getBlock()->printAsOperand(llvm::dbgs());
                            llvm::dbgs() << '\n';
                        });
                        queue.emplace(std::move(iter), front.second);
                    }
                    continue;
                }

                // Orchestrator returned without a successor and might have finished function execution of the whole
                // function.
                LLVM_DEBUG({ llvm::dbgs() << "Finished with no successors\n"; });
                auto* orch = front.second.get();
                if (front.second.release() != 0)
                {
                    continue;
                }

                if (orch->inCalls())
                {
                    // May not be queued up anymore but there are still calls we are waiting for.
                    continue;
                }

                // Truly finished case.
                LLVM_DEBUG({ llvm::dbgs() << "Finished function execution: Retiring\n"; });
                scheduleWaitingCalls(queue, orch);
                orch->retire();
                continue;
            }
            auto& call = pylir::get<FunctionCall>(result);
            LLVM_DEBUG({ llvm::dbgs() << "Suspended with call to " << call.functionSpecialization << '\n'; });

            if (call.functionSpecialization.function.isExternal()
                || llvm::any_of(call.functionSpecialization.argTypes,
                                [this](TypeFlowArgValue arg)
                                {
                                    if (auto type = arg.dyn_cast<pylir::Py::ObjectTypeInterface>())
                                    {
                                        return typeIsOverDefined(type);
                                    }
                                    return false;
                                }))
            {
                if (!call.functionSpecialization.function.isExternal())
                {
                    LLVM_DEBUG({
                        llvm::dbgs() << "Function argument contains more than " << m_maxTypeSize
                                     << " types. Function call marked over defined.\n";
                    });
                    m_typesOverDefined++;
                }
                else
                {
                    LLVM_DEBUG({ llvm::dbgs() << "Calling external function. Function call marked over defined.\n"; });
                }
                for (auto dest : call.resultValues)
                {
                    front.first.getValues()[dest] = {};
                }
                queue.emplace(std::move(front));
                continue;
            }

            auto [existing, inserted] = m_orchestrators.insert({std::move(call.functionSpecialization), nullptr});
            if (inserted)
            {
                LLVM_DEBUG({ llvm::dbgs() << "First time calling\n"; });
                existing->second = createOrchestrator(existing->first.function, moduleManager);
            }
            else if (existing->second->finishedExecution())
            {
                LLVM_DEBUG({ llvm::dbgs() << "Already finished\n"; });
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
                LLVM_DEBUG({
                    if (!inserted)
                    {
                        llvm::dbgs() << "Still executing: Adding to waiters\n";
                    }
                });
                existing->second->addWaitingCall(std::move(front.first), orch, call.resultValues);
                auto handleRecursion = [&]
                {
                    LLVM_DEBUG({ llvm::dbgs() << "Recursion detected: Marking function call over defined\n"; });
                    for (auto dest : call.resultValues)
                    {
                        front.first.getValues()[dest] = {};
                    }
                    queue.emplace(std::move(front));
                };
                if (orch != existing->second.get())
                {
                    if (auto cycle = addEdge(orch, existing->second.get()); !cycle.empty())
                    {
                        handleRecursion();
                        continue;
                    }
                }
                else
                {
                    handleRecursion();
                    continue;
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
                continue;
            }

            // First call, set up the function arguments.
            llvm::DenseMap<mlir::Value, pylir::Py::TypeAttrUnion> entryValues;
            for (auto [arg, value] :
                 llvm::zip(existing->second->getEntryBlock()->getArguments(), existing->first.argTypes))
            {
                if (auto ref = value.dyn_cast<pylir::Py::RefAttr>())
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

    std::vector<std::pair<FunctionSpecialization, std::unique_ptr<Orchestrator>>> moveResults() &&
    {
        auto vec = m_orchestrators.takeVector();
        auto range = llvm::make_range(std::move_iterator(vec.begin()), std::move_iterator(vec.end()));
        return {range.begin(), range.end()};
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

    std::vector<std::pair<FunctionSpecialization, std::unique_ptr<Orchestrator>>> results;
    {
        Scheduler scheduler(m_maxTypeSize, m_typesOverDefined);
        scheduler.run(roots.getArrayRef(), getAnalysisManager());
        results = std::move(scheduler).moveResults();
    }

    struct Clone
    {
        mlir::FunctionOpInterface function;
        mlir::BlockAndValueMapping mapping;
    };

    bool changed = false;
    mlir::SymbolTable table(getOperation());

    auto doClone = [this, &table](Clone& clone)
    {
        clone.function = clone.function->clone(clone.mapping);
        mlir::cast<mlir::SymbolOpInterface>(*clone.function).setPrivate();
        table.insert(clone.function);
        m_functionsCloned++;
    };

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
                instrValue = mapping.mapResult(key);
            }

            // Cloning of a function body is done lazily for the case where no value has changed.
            // Roots are not cloned but updated in place.
            if (clone.function == func.function && !isRoot)
            {
                doClone(clone);
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
            // Due to the lack of a better way we just use PylirPyDialect for now. It can handle every kind of constant
            // including ones by arith.
            mlir::Dialect* dialect = getContext().getLoadedDialect<pylir::Py::PylirPyDialect>();
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
                if (!func.function || !calleeClone.function || calleeClone.function == func.function
                    || !calleeDiffers(call, mlir::FlatSymbolRefAttr::get(calleeClone.function)))
                {
                    continue;
                }

                // Lazy cloning of this function if it has not yet occurred. Just like in the constant setting loop.
                if (thisClone.function == thisFunc.function && !isRoot)
                {
                    doClone(thisClone);
                    cloneOccurred = true;
                    call = calcCall();
                }

                auto* newCall = setCallee(call, mlir::FlatSymbolRefAttr::get(calleeClone.function));
                if (newCall && thisClone.function != thisFunc.function)
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
