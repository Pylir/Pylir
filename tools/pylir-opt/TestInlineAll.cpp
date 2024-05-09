//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Pass/PassManager.h>

#include <pylir/Optimizer/PylirPy/Transforms/Util/InlinerUtil.hpp>

#include "Passes.hpp"

namespace pylir::test {
#define GEN_PASS_DEF_TESTINLINEALLPASS
#include "Passes.h.inc"
} // namespace pylir::test

namespace {
class RecursionStateMachine {
  std::vector<mlir::CallableOpInterface> m_pattern;
  decltype(m_pattern)::const_iterator m_inPattern;
  std::size_t m_count = 1;

  enum class States {
    GatheringPattern,
    Synchronizing,
    CheckingPattern,
  };

  States m_state = States::GatheringPattern;

public:
  explicit RecursionStateMachine(mlir::CallableOpInterface firstOccurrence)
      : m_pattern{firstOccurrence} {}

  ~RecursionStateMachine() = default;
  RecursionStateMachine(const RecursionStateMachine&) = delete;
  RecursionStateMachine& operator=(const RecursionStateMachine&) = delete;
  RecursionStateMachine(RecursionStateMachine&&) noexcept = default;
  RecursionStateMachine& operator=(RecursionStateMachine&&) noexcept = default;

  [[nodiscard]] std::size_t getCount() const {
    return m_count;
  }

  [[nodiscard]] llvm::ArrayRef<mlir::CallableOpInterface> getPattern() const {
    return m_pattern;
  }

  void cycle(mlir::CallableOpInterface occurrence) {
    switch (m_state) {
    case States::Synchronizing:
      if (occurrence != m_pattern.front()) {
        return;
      }
      m_state = States::GatheringPattern;
      break;
    case States::GatheringPattern:
      if (occurrence != m_pattern.front()) {
        m_pattern.push_back(occurrence);
        return;
      }
      m_state = States::CheckingPattern;
      m_inPattern = m_pattern.begin() + (m_pattern.size() == 1 ? 0 : 1);
      m_count = 2;
      break;
    case States::CheckingPattern:
      if (*m_inPattern == occurrence) {
        if (++m_inPattern == m_pattern.end()) {
          m_count++;
          m_inPattern = m_pattern.begin();
        }
        return;
      }
      m_state = occurrence == m_pattern.front() ? States::GatheringPattern
                                                : States::Synchronizing;
      m_count = 0;
      m_pattern.resize(1);
      break;
    }
  }
};

class TestInlineAll
    : public pylir::test::impl::TestInlineAllPassBase<TestInlineAll> {
  mlir::OpPassManager m_passManager;

public:
  using Base::Base;

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    Base::getDependentDialects(registry);
    mlir::OpPassManager temp;
    if (mlir::failed(mlir::parsePassPipeline(m_optimizationPipeline, temp,
                                             llvm::nulls())))
      return;

    temp.getDependentDialects(registry);
  }

protected:
  mlir::LogicalResult initialize(mlir::MLIRContext*) override {
    return mlir::parsePassPipeline(m_optimizationPipeline, m_passManager);
  }

  void runOnOperation() override {
    auto* init = getOperation().lookupSymbol("__init__");
    if (!init) {
      llvm::errs() << "__init__ symbol required\n";
      signalPassFailure();
      return;
    }

    llvm::MapVector<mlir::CallableOpInterface, RecursionStateMachine>
        recursionDetection;
    llvm::DenseSet<mlir::CallableOpInterface> disabledCallables;
    auto recursivePattern = [&](mlir::CallableOpInterface callable) {
      bool triggered = false;
      for (auto* iter = recursionDetection.begin();
           iter != recursionDetection.end();) {
        auto& stateMachine = iter->second;
        stateMachine.cycle(callable);
        if (stateMachine.getCount() >= m_maxRecursionDepth) {
          auto pattern = stateMachine.getPattern();
          disabledCallables.insert(pattern.begin(), pattern.end());
          iter = recursionDetection.erase(iter);
          triggered = true;
          m_recursionLimitReached++;
          continue;
        }
        iter++;
      }
      if (!triggered) {
        // If it was triggered, then this callable was the fist occurrence in a
        // recursion pattern and now disabled and deleted. Don't reinsert it now
        recursionDetection.insert({callable, RecursionStateMachine{callable}});
      }
    };

    mlir::SymbolTableCollection collection;
    bool changed = false;
    do {
      bool failed = false;
      changed = false;
      init->walk([&](mlir::CallOpInterface callOpInterface) {
        auto ref = mlir::dyn_cast<mlir::SymbolRefAttr>(
            callOpInterface.getCallableForCallee());
        if (!ref) {
          return mlir::WalkResult::advance();
        }
        auto callable =
            collection.lookupNearestSymbolFrom<mlir::CallableOpInterface>(
                callOpInterface, ref);
        if (!callable) {
          return mlir::WalkResult::advance();
        }
        if (disabledCallables.contains(callable)) {
          return mlir::WalkResult::advance();
        }
        pylir::Py::inlineCall(callOpInterface, callable);
        m_callsInlined++;
        changed = true;
        if (mlir::failed(runPipeline(m_passManager, init))) {
          failed = true;
        }
        recursivePattern(callable);
        return mlir::WalkResult::interrupt();
      });
      if (failed) {
        signalPassFailure();
        return;
      }
    } while (changed);
  }
};
} // namespace
