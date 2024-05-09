//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SparseBitVector.h>
#include <llvm/Support/Debug.h>

#include <pylir/Optimizer/Analysis/InlineCost.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/Value.hpp>
#include <pylir/Support/Macros.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>

#include "Passes.hpp"
#include "Util/InlinerUtil.hpp"

using namespace mlir;
using namespace pylir::Py;

#define DEBUG_TYPE "inliner"

namespace pylir::Py {
#define GEN_PASS_DEF_INLINERPASS
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"
} // namespace pylir::Py

namespace {

using namespace pylir;

class InlinerPass : public Py::impl::InlinerPassBase<InlinerPass> {
  mlir::OpPassManager m_passManager;

protected:
  mlir::LogicalResult initialize(mlir::MLIRContext*) override {
    auto temp = mlir::parsePassPipeline(m_optimizationPipeline);
    if (mlir::failed(temp))
      return temp;

    m_passManager = std::move(*temp);
    return mlir::success();
  }

  void runOnOperation() override;

public:
  using Base::Base;

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    Base::getDependentDialects(registry);
    // Above initialize will signal the error properly. This also gets called
    // before `initialize`, hence we can't use m_passManager here.
    auto temp = mlir::parsePassPipeline(m_optimizationPipeline, llvm::nulls());
    if (mlir::failed(temp))
      return;

    temp->getDependentDialects(registry);
  }
};

class CallSite {
  // Call op of the call-site.
  mlir::CallOpInterface m_call;
  // Callable callee.
  mlir::CallableOpInterface m_callee;
  // Further callsites reachable if this callsite were to be inlined.
  std::vector<std::pair<mlir::CallOpInterface, mlir::CallableOpInterface>>
      m_reachableCallsites;
  // Marks the callsite as erased, causing the queue to skip and delete it.
  bool m_erased = false;
  // Cost of inlining the callee into the call in abstract units.
  std::uint16_t m_cost{};

  friend class CallSiteQueue;

public:
  CallSite(
      mlir::CallOpInterface call, mlir::CallableOpInterface callee,
      std::uint16_t cost,
      std::vector<std::pair<mlir::CallOpInterface, mlir::CallableOpInterface>>&&
          reachableCallsites)
      : m_call(call), m_callee(callee),
        m_reachableCallsites(std::move(reachableCallsites)), m_cost(cost) {}

  /// Returns the call op of the callsite.
  [[nodiscard]] mlir::CallOpInterface getCall() const {
    return m_call;
  }

  /// Returns the callee of the callsite.
  [[nodiscard]] mlir::CallableOpInterface getCallee() const {
    return m_callee;
  }

  /// Returns the cost of inlining the callee into the callsite, as estimated by
  /// the heuristic, in abstract units.
  [[nodiscard]] std::size_t getCost() const {
    return m_cost;
  }

  /// Returns the list of callsites that are reachable after this callsite has
  /// been inlined.
  llvm::ArrayRef<std::pair<mlir::CallOpInterface, mlir::CallableOpInterface>>
  getReachableCallsites() const {
    return m_reachableCallsites;
  }

  std::vector<std::pair<mlir::CallOpInterface, mlir::CallableOpInterface>>&
  getReachableCallsites() {
    return m_reachableCallsites;
  }

  /// Marks the callsite as erased, effectively removing it from any public APIs
  /// of the queue.
  void erase() {
    m_erased = true;
  }
};

/// Queue of callsites, determining the order of callsites being inlined.
/// The implementation is currently simply a min-heap over the cost of
/// callsites.
///
/// Calling 'erase' on a Callsite effectively removes it from the queue. Due to
/// the difficulty of erasing an element from a heap, it is done so in a lazy
/// fashion. That is, the callsite is just marked as erased and when popping a
/// callsite marked erased from the minheap, it is simply skipped and the next
/// element is taken from the min-heap.
///
/// Additionally, the queue tracks all callers of a callable and allows their
/// retrieval. This information is also kept up to date when erasing any
/// callsites.
class CallSiteQueue {
  // Allocator for all CallSites. Improves locality and simplifies lifetime
  // management.
  llvm::SpecificBumpPtrAllocator<CallSite> m_allocator;
  // Min-heap.
  std::vector<CallSite*> m_queue;
  // Actually CallableOpInterface as keys, but using Operation* saves us on
  // storing one pointer per key-value pair.
  llvm::DenseMap<mlir::Operation*, std::vector<CallSite*>> m_callers;

  // Comparator used to turn, the default max-heap implementation into a
  // min-heap implementation.
  struct CallSiteComp {
    bool operator()(decltype(m_queue)::const_reference lhs,
                    decltype(m_queue)::const_reference rhs) const noexcept {
      return lhs->m_cost > rhs->m_cost;
    }
  };

public:
  /// Constructs a new CallSite from the given arguments and adds it to the
  /// queue.
  template <class... Args>
  void emplace(Args&&... args) {
    m_queue.push_back(new (m_allocator.Allocate())
                          CallSite(std::forward<Args>(args)...));
    m_callers[m_queue.back()->getCallee()].push_back(m_queue.back());
    std::push_heap(m_queue.begin(), m_queue.end(), CallSiteComp{});
  }

  /// Clears the queue, resulting in it deallocating all memory and deleteing
  /// all callsites.
  void clear() {
    m_queue.clear();
    m_callers.clear();
    m_allocator.DestroyAll();
  }

  /// Returns all current call-sites calling the callable within the queue.
  llvm::ArrayRef<CallSite*> getCallers(mlir::CallableOpInterface callable) {
    auto iter = m_callers.find(callable);
    if (iter == m_callers.end())
      return {};

    // Lazily delete all callsites that have been marked as erased while we're
    // at it.
    llvm::erase_if(iter->second,
                   [](CallSite* callSite) { return callSite->m_erased; });
    return iter->second;
  }

  /// Returns a pointer to the callsite that should be inlined next and deletes
  /// it from the queue. Returns a nullptr if the queue is empty. The lifetime
  /// of the callsite is equal to the lifetime of the queue or until 'clear' is
  /// called.
  const CallSite* pop() {
    while (!m_queue.empty()) {
      std::pop_heap(m_queue.begin(), m_queue.end(), CallSiteComp{});
      CallSite* callSite = m_queue.back();
      m_queue.pop_back();
      if (callSite->m_erased)
        continue;

      callSite->m_erased = true;
      return callSite;
    }
    return nullptr;
  }
};

struct GradeResult {
  std::uint16_t cost;
  std::vector<std::pair<mlir::CallOpInterface, mlir::CallableOpInterface>>
      reachableCallsites;
};

/// Class used for managing the inline history of an edge. For every edge,
/// identified by the 'py.edge_id' attribute, it keeps a map of callables that
/// have been inlined through that edge and how many times such a callable was
/// inlined through that edge.
class EdgeHistories {
  // We use a free-list of maps to re-use the storage of dead maps across
  // inlining iterations. For that purpose, the map itself, with a single entry
  // is actually used to hold the index of the next free cell. The size of 2
  // here was carefully chosen, as it allows for that single entry to be
  // allocated inline.
  using Map = llvm::SmallDenseMap<mlir::StringAttr, std::size_t, 2>;

  std::vector<Map> m_histories;
  std::size_t m_nextFree = std::numeric_limits<std::size_t>::max();

  void markNextFree(std::size_t index) {
    Map& map = m_histories[index];
    // Doing this rather than shrink_and_fit leads to the de-allocation of
    // buckets which we desire. Keeping the buckets is useless in terms of
    // avoiding allocations, because both the copy and move assignment operators
    // of maps do not reuse the current storage.
    map = Map();
    map[nullptr] = m_nextFree;
    m_nextFree = index;
  }

public:
  /// Adds a new edge to the history. 'call' is assumed to be a call-site that
  /// was created during the inlining of 'originatingCallee'. The call is given
  /// a unique 'py.edge_id' and 'originatingCallee' is added to the map of
  /// callables inlined through 'call'. If 'call' already had a history/edge id,
  /// the initial history is taken from the existing.
  void addNewEntry(mlir::CallOpInterface call,
                   mlir::CallableOpInterface originatingCallee) {
    Map existing;
    if (auto existingId = call->getAttrOfType<mlir::IntegerAttr>("py.edge_id"))
      existing = m_histories[existingId.getValue().getZExtValue()];

    std::size_t id;
    if (m_nextFree == std::numeric_limits<std::size_t>::max()) {
      id = m_histories.size();
      m_histories.push_back(std::move(existing));
    } else {
      id = m_nextFree;
      m_nextFree = m_histories[id].begin()->second;
      m_histories[id] = std::move(existing);
    }

    call->setAttr("py.edge_id",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(call->getContext(), 32), id));
    m_histories[id][mlir::SymbolTable::getSymbolName(originatingCallee)]++;
  }

  /// Returns how many times 'callee' has been inlined through 'call'
  /// transitively.
  std::size_t getNumOccurrences(mlir::CallOpInterface call,
                                mlir::CallableOpInterface callee) const {
    auto integer = call->getAttrOfType<mlir::IntegerAttr>("py.edge_id");
    if (!integer)
      return 0;

    return m_histories[integer.getValue().getZExtValue()].lookup(
        mlir::SymbolTable::getSymbolName(callee.getOperation()));
  }

  /// Returns the amount of edges whose histories have been added.
  std::size_t size() const {
    return m_histories.size();
  }

  /// Erases the entries of all edges whose bit corresponding to its id has been
  /// set in the vector.
  void erase(const llvm::BitVector& vector) {
    for (std::size_t i = 0;
         i < std::min<std::size_t>(m_histories.size(), vector.size()); i++) {
      if (!vector.test(i))
        continue;

      markNextFree(i);
    }
  }
};

class Inliner {
  mlir::ModuleOp m_module;
  // Maximum cost a callsite may have to still be inlined.
  std::uint16_t m_threshold;
  std::uint16_t m_cyclePenalty;
  mlir::AnalysisManager m_analysisManager;
  CallSiteQueue m_queue;

  EdgeHistories m_edgeHistories;

  // Compute the inline cost estimate using of the call to the callee.
  [[nodiscard]] std::optional<GradeResult>
  grade(mlir::CallOpInterface call, mlir::CallableOpInterface callee,
        mlir::SymbolTableCollection& collection);

  void pruneEgeHistory() {
    llvm::BitVector dead(m_edgeHistories.size(), true);
    m_module->walk([&](mlir::Operation* op) {
      auto attr = op->getAttrOfType<mlir::IntegerAttr>("py.edge_id");
      if (!attr || attr.getValue().uge(dead.size()))
        return;

      dead.reset(attr.getValue().getZExtValue());
    });
    m_edgeHistories.erase(dead);
  }

public:
  Inliner(mlir::ModuleOp moduleOp, std::uint16_t threshold,
          std::uint16_t cyclePenalty, mlir::AnalysisManager analysisManager)
      : m_module(moduleOp), m_threshold(threshold),
        m_cyclePenalty(cyclePenalty), m_analysisManager(analysisManager) {}

  /// Performs one iteration of inlining on the module.
  /// Returns true if at least one callsite was inlined.
  bool performInlining(mlir::Pass::Statistic& callsInlined,
                       mlir::Pass::Statistic& directRecursionsDiscarded,
                       mlir::Pass::Statistic& callsitesTooExpensive);
};

mlir::Operation* getNextClosestIsolatedFromAbove(mlir::Operation* op) {
  if (op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
    return op;

  return op->getParentWithTrait<mlir::OpTrait::IsIsolatedFromAbove>();
}

mlir::CallableOpInterface
deduceCallableFromCall(mlir::CallOpInterface call,
                       mlir::SymbolTableCollection& collection) {
  // Try our best to get a CallableOpInterface out of the call. For direct calls
  // this is simply looking up the symbol in the symbol table, but for indirect
  // calls we can only check whether the indirect callee is an actual constant
  // or possibly even the callable op itself!
  mlir::CallInterfaceCallable callee = call.getCallableForCallee();
  auto ref = dyn_cast<mlir::SymbolRefAttr>(callee);
  if (!ref)
    mlir::matchPattern(callee.get<mlir::Value>(), mlir::m_Constant(&ref));

  mlir::CallableOpInterface callable;
  if (ref) {
    callable = collection.lookupNearestSymbolFrom<mlir::CallableOpInterface>(
        call, ref);
  } else {
    callable =
        callee.get<mlir::Value>().getDefiningOp<mlir::CallableOpInterface>();
  }
  return callable;
}

/// Estimate the inline cost of 'code' given the values in 'knownConstants' are
/// of the given constant values. If the cost is larger than 'threshold', the
/// estimation calculation is cut short and a nullopt is returned.
std::optional<GradeResult> gradeFromKnownConstants(
    llvm::DenseMap<mlir::Value, mlir::Attribute>&& knownConstants,
    mlir::CallableOpInterface code, std::uint16_t threshold,
    mlir::SymbolTableCollection& collection) {
  llvm::DenseSet<mlir::Block*> liveBlocks;

  std::vector<std::pair<mlir::CallOpInterface, mlir::CallableOpInterface>>
      reachableCallsites;

  InlineCost bodySize(code->getContext());

  // Simple implementation that simply goes through all reachable blocks and
  // sums up the inline costs of each of their operations. Given initial
  // constant values given through 'knownConstants' we first attempt to constant
  // fold all operations. If we succeed, the operations cost is determined as 0,
  // and its resulting values are added to 'knownConstants'. If we hit a
  // conditional branching operation, we attempt to resolve the branching
  // result, allowing us to only mark the given successor as live, instead of
  // all successors.
  std::uint16_t cost = 0;
  code->walk<mlir::WalkOrder::PreOrder>([&](mlir::Block* block)
                                            -> mlir::WalkResult {
    if (cost > threshold)
      return mlir::WalkResult::interrupt();

    // Entry block of a region is always live, and should be added to the set.
    if (block->isEntryBlock())
      liveBlocks.insert(block);
    else if (!liveBlocks.contains(block))
      // If the block is not live we skip it and any contained ops and regions.
      return mlir::WalkResult::skip();

    for (mlir::Operation& op : *block) {
      // Early exit if the cost is already higher than the threshold.
      // There is no point in continuing.
      if (cost > threshold)
        return mlir::WalkResult::interrupt();

      // Special case for ops, that been constant-folded in the caller and
      // passed in via 'knownConstants'. We'd otherwise add their size.
      if (op.getNumResults() == 1 && knownConstants.count(op.getResult(0)))
        continue;

      if (auto call = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
        auto callable = deduceCallableFromCall(call, collection);
        if (!callable) {
          // TODO: All of this is in need of an interface function or something
          // not as hardcoded.
          if (auto constant = knownConstants.lookup(
                  dyn_cast<mlir::Value>(call.getCallableForCallee()))) {
            if (auto func = dyn_cast<FunctionAttrInterface>(constant)) {
              callable =
                  collection.lookupNearestSymbolFrom<mlir::CallableOpInterface>(
                      call, func.getValue());
            }
            if (auto ref = mlir::dyn_cast<mlir::SymbolRefAttr>(constant)) {
              callable =
                  collection.lookupNearestSymbolFrom<mlir::CallableOpInterface>(
                      call, ref);
            }
          }
        }
        if (callable)
          reachableCallsites.emplace_back(call, callable);
      }

      auto constantOperands = llvm::to_vector(
          llvm::map_range(op.getOperands(), [&](mlir::Value value) {
            return knownConstants.lookup(value);
          }));
      if (&op != block->getTerminator()) {
        // Have to save both the operands and attributes, as a fold operation
        // can inplace modify the operation. This would be wrong here in the
        // case of the inlining not actually happening.
        auto operandsPrior = llvm::to_vector(op.getOperands());
        mlir::DictionaryAttr attrPrior = op.getAttrDictionary();

        llvm::SmallVector<mlir::OpFoldResult> result;
        if (mlir::failed(op.fold(constantOperands, result))) {
          cost += bodySize.getCostOf(&op);
          continue;
        }
        op.setOperands(operandsPrior);
        op.setAttrs(attrPrior);

        for (auto [foldRes, res] : llvm::zip(result, op.getResults())) {
          if (!foldRes)
            continue;

          if (foldRes.is<mlir::Attribute>()) {
            knownConstants.insert({res, foldRes.get<mlir::Attribute>()});
            continue;
          }

          if (mlir::Attribute constantValue =
                  knownConstants.lookup(foldRes.get<mlir::Value>()))
            knownConstants.insert({res, constantValue});
        }
        continue;
      }

      auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(op);
      if (!branch) {
        cost += bodySize.getCostOf(&op);
        break;
      }

      mlir::Block* successor = branch.getSuccessorForOperands(constantOperands);
      for (std::size_t succIndex :
           llvm::seq<std::size_t>(0, op.getNumSuccessors())) {
        mlir::Block* succ = op.getSuccessor(succIndex);
        // Don't propagate constant operands to the block arguments of the
        // successor if it is not the only predecessor it has. Note that we
        // can't check for the liveness of predecessor blocks, as they may not
        // yet have been visited, or it might be part of a loop and therefore
        // simply cannot have been visited yet, even if the successor were to be
        // visited.
        if ((successor && succ != successor) || !succ->getSinglePredecessor())
          continue;

        mlir::SuccessorOperands operands =
            branch.getSuccessorOperands(succIndex);
        for (auto [blockArg, value] :
             llvm::zip(succ->getArguments().drop_front(
                           operands.getProducedOperandCount()),
                       operands.getForwardedOperands()))
          if (mlir::Attribute constantValue = knownConstants.lookup(value))
            knownConstants.insert({blockArg, constantValue});
      }

      if (successor) {
        liveBlocks.insert(successor);
      } else {
        cost += bodySize.getCostOf(branch);
        liveBlocks.insert(op.successor_begin(), op.successor_end());
      }
    }

    return mlir::WalkResult::advance();
  });
  if (cost > threshold)
    return std::nullopt;

  return GradeResult{cost, std::move(reachableCallsites)};
}

std::optional<GradeResult>
Inliner::grade(mlir::CallOpInterface call, mlir::CallableOpInterface callee,
               mlir::SymbolTableCollection& collection) {
  llvm::DenseMap<mlir::Value, mlir::Attribute> knownConstants;
  mlir::Region* region = callee.getCallableRegion();
  for (auto [arg, parameter] :
       llvm::zip(call.getArgOperands(), region->getArguments())) {
    mlir::Attribute res;
    if (mlir::matchPattern(arg, mlir::m_Constant(&res))) {
      knownConstants.insert({parameter, res});
    } else if (auto type = Py::getTypeOf(arg);
               type && type.is<mlir::Attribute>()) {
      for (mlir::Operation* user : parameter.getUsers()) {
        auto typeOf = mlir::dyn_cast<Py::TypeOfOp>(user);
        if (!typeOf)
          continue;

        knownConstants.insert({typeOf, type.get<mlir::Attribute>()});
      }
    }
  }

  std::int32_t threshold =
      m_threshold -
      m_cyclePenalty * static_cast<std::ptrdiff_t>(
                           m_edgeHistories.getNumOccurrences(call, callee));
  if (threshold < 0)
    return std::nullopt;

  return gradeFromKnownConstants(std::move(knownConstants), callee, threshold,
                                 collection);
}

// Formatting function for LLVM_DEBUG output.
[[maybe_unused]] std::string
formatCalleeForDebug(mlir::CallableOpInterface callableOpInterface) {
  std::string result;
  mlir::SymbolOpInterface symbol;
  while (true) {
    symbol = mlir::dyn_cast<mlir::SymbolOpInterface>(*callableOpInterface);
    if (symbol)
      break;

    result =
        ("::" + callableOpInterface->getName().getStringRef() + result).str();
    callableOpInterface =
        callableOpInterface->getParentOfType<mlir::CallableOpInterface>();
  }
  return (symbol.getName() + result).str();
}

[[maybe_unused]] std::string
formatCallerForDebug(mlir::CallOpInterface callOpInterface) {
  return formatCalleeForDebug(
      callOpInterface->getParentOfType<mlir::CallableOpInterface>());
}

bool Inliner::performInlining(mlir::Pass::Statistic& callsInlined,
                              mlir::Pass::Statistic& directRecursionsDiscarded,
                              mlir::Pass::Statistic& callsitesTooExpensive) {
  pruneEgeHistory();

  mlir::SymbolTableCollection collection;

  auto handleCallOp = [&](mlir::CallOpInterface call,
                          mlir::CallableOpInterface callable) {
    // A callable without a callable region can't be inlined as it has no body.
    // This is the case for function declarations for example.
    if (!callable.getCallableRegion())
      return;

    std::optional<GradeResult> grading = grade(call, callable, collection);
    if (!grading) {
      // TODO: Should these call-sites be added to the caller list tracked by
      // the queue anyways? That would allow
      //  these callsites to be reevaluated after inlining changes have
      //  happened. At the same time, inlining operations only worsen the grade
      //  99% of the time, and improvements are usually only visible after the
      //  interleaved simplification passes have run, in which case all
      //  call-sites are regraded anyways.
      callsitesTooExpensive++;
      return;
    }

    // If this is a directly recursive callsite and a recursive callsite remains
    // after inlining, don't inline. Recursive functions must be handled
    // specially, else-way they easily lead to exponential code explosion.
    if (callable->isAncestor(call) &&
        llvm::is_contained(llvm::make_second_range(grading->reachableCallsites),
                           callable)) {
      directRecursionsDiscarded++;
      return;
    }

    m_queue.emplace(call, callable, grading->cost,
                    std::move(grading->reachableCallsites));
  };

  m_module.walk([&](mlir::CallOpInterface callOpInterface) {
    mlir::CallableOpInterface callable =
        deduceCallableFromCall(callOpInterface, collection);
    if (!callable)
      return;

    handleCallOp(callOpInterface, callable);
  });

  bool changed = false;
  while (const CallSite* callSite = m_queue.pop()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Inlining " << formatCalleeForDebug(callSite->getCallee())
                   << " into " << formatCallerForDebug(callSite->getCall())
                   << " with cost " << callSite->getCost() << '\n';
    });

    callsInlined++;
    changed = true;

    mlir::CallableOpInterface callerCallable;
    for (mlir::Operation* curr = callSite->getCall(); !callerCallable && curr;
         curr = curr->getParentOp()) {
      callerCallable = mlir::dyn_cast<mlir::CallableOpInterface>(curr);
    }
    PYLIR_ASSERT(callerCallable && "Every callsite must be within a callable");

    mlir::Operation* closestIsolatedFromAbove =
        getNextClosestIsolatedFromAbove(callSite->getCall());
    mlir::IRMapping mapping =
        Py::inlineCall(callSite->getCall(), callSite->getCallee());

    // Following the inlining we have to invalidate all analysis' dependent on
    // the structure of the IR of the caller. MLIRs infrastructure currently has
    // no such fine-grained framework of knowing which analysis' should be
    // invalidated when, therefore we conservatively invalidate them all. Note:
    // code currently "wrongly" assumes just a two-level nesting of
    // isolated-from-above ops (functions in modules).
    m_analysisManager.nest(closestIsolatedFromAbove)
        .invalidate(mlir::AnalysisManager::PreservedAnalyses());

    // Use to_vector to make a local copy of the callers. This is necessary to
    // prevent a use-after-free error if we are emplacing into the queue below
    // and that'd lead to adding a new caller to the callable.
    for (CallSite* existing :
         llvm::to_vector(m_queue.getCallers(callerCallable))) {
      // If the score remains the same, the callsite, or rather its position in
      // the queue, is not out-of-date and there is no need to replace it with a
      // new one.
      auto grading =
          grade(existing->getCall(), existing->getCallee(), collection);
      if (grading && grading->cost == existing->getCost())
        continue;

      if (grading)
        m_queue.emplace(existing->getCall(), existing->getCallee(),
                        grading->cost,
                        std::move(existing->getReachableCallsites()));
      else
        callsitesTooExpensive++;

      existing->erase();
    }

    // Note that the callsite call op is not the inlined callsite, but the
    // operation within the original callee. We use the IRMapper returned by the
    // inlining function to map it to the inlined op.
    for (auto [sourceCall, callable] : callSite->getReachableCallsites()) {
      auto newCall = mlir::dyn_cast_or_null<mlir::CallOpInterface>(
          mapping.lookupOrNull(sourceCall.getOperation()));
      if (!newCall)
        continue;

      m_edgeHistories.addNewEntry(newCall, callSite->getCallee());
      handleCallOp(newCall, callable);
    }
  }
  return changed;
}

void InlinerPass::runOnOperation() {
  // Initial simplification before any inlining even starts.
  m_optimizationRun++;
  if (mlir::failed(runPipeline(m_passManager, getOperation()))) {
    signalPassFailure();
    return;
  }

  Inliner inliner(getOperation(), m_threshold, m_cyclePenalty,
                  getAnalysisManager());
  bool changed = false;
  [[maybe_unused]] bool escapedEarly = false;
  for (std::size_t i = 0; i < m_maxInliningIterations; i++) {
    LLVM_DEBUG({ llvm::dbgs() << "Inlining iteration " << i << '\n'; });
    if (!inliner.performInlining(m_callsInlined, m_directRecursionsDiscarded,
                                 m_callsitesTooExpensive)) {
      m_doneEarly++;
      escapedEarly = true;
      break;
    }

    // The optimization pipeline is run over the whole module since it contains
    // module level passes. If it were just function passes we could only run
    // optimizations on any functions that changed through inlining.
    changed = true;
    m_optimizationRun++;
    if (mlir::failed(runPipeline(m_passManager, getOperation()))) {
      signalPassFailure();
      return;
    }
  }

  LLVM_DEBUG({
    if (escapedEarly) {
      llvm::dbgs() << "Quit inlining due to no more changes\n";
    } else {
      llvm::dbgs() << "Quit inlining due iteration limit reached\n";
    }
  });

  // Get rid of the intermediate id for nicer output.
  getOperation()->walk(
      [&](mlir::Operation* op) { op->removeAttr("py.edge_id"); });

  if (changed)
    markAllAnalysesPreserved();
}
} // namespace
