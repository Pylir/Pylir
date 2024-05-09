//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

#include <pylir/Optimizer/Analysis/AliasSetTracker.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyTraits.hpp>

#include "Passes.hpp"

namespace pylir::test {
#define GEN_PASS_DEF_TESTALIASSETTRACKERPASS
#include "Passes.h.inc"
} // namespace pylir::test

namespace {
class TestAliasSetTracker
    : public pylir::test::impl::TestAliasSetTrackerPassBase<
          TestAliasSetTracker> {
protected:
  void runOnOperation() override {
    for (auto iter : getOperation().getOps<mlir::FunctionOpInterface>()) {
      auto& aliasAnalysis = getChildAnalysis<mlir::AliasAnalysis>(iter);
      pylir::AliasSetTracker tracker(aliasAnalysis);
      iter.walk([&](mlir::Operation* op) {
        if (op == &op->getBlock()->front()) {
          for (auto& iter2 : op->getBlock()->getArguments()) {
            if (!mlir::isa<pylir::Py::DynamicType>(iter2.getType()))
              continue;
            tracker.insert(iter2);
          }
        }
        if (op->hasTrait<pylir::Py::ReturnsImmutable>())
          return;

        for (auto res : op->getResults()) {
          if (!mlir::isa<pylir::Py::DynamicType>(res.getType()))
            continue;
          tracker.insert(res);
        }
      });

      auto state = mlir::AsmState(iter);
      llvm::outs() << "Alias sets for " << iter.getName() << ":\n";
      for (const auto& iter2 : tracker) {
        llvm::outs() << "{";
        for (auto iter3 : iter2) {
          iter3.printAsOperand(llvm::outs(), state);
          llvm::outs() << " ";
        }
        llvm::outs() << "}\n";
      }
    }
  }

public:
  using Base::Base;
};
} // namespace
