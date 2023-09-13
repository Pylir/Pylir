// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/AsmState.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/FunctionInterfaces.h>

#include <pylir/Optimizer/Analysis/EscapeAnalysis.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyTypes.hpp>

#include "Passes.hpp"

namespace pylir::test {
#define GEN_PASS_DEF_TESTESCAPEANALYSISPASS
#include "Passes.h.inc"
} // namespace pylir::test

namespace {
class TestEscapeAnalysisPass
    : public pylir::test::impl::TestEscapeAnalysisPassBase<
          TestEscapeAnalysisPass> {
protected:
  void runOnOperation() override {
    for (auto iter : getOperation().getOps<mlir::FunctionOpInterface>()) {
      auto& analysis = getChildAnalysis<pylir::EscapeAnalysis>(iter);

      // Print the function first, allowing us to capture SSA names in
      // FileCheck.
      iter->print(llvm::errs());
      llvm::errs() << "\nEscapes: ";
      bool first = true;
      mlir::AsmState state(iter);
      iter.getFunctionBody().walk([&](mlir::Operation* op) {
        for (mlir::Value result :
             llvm::make_filter_range(op->getResults(), [](mlir::Value v) {
               return mlir::isa<pylir::Py::DynamicType>(v.getType());
             })) {
          if (!analysis.escapes(result))
            continue;
          if (first) {
            first = false;
          } else {
            llvm::errs() << ", ";
          }
          result.printAsOperand(llvm::errs(), state);
        }
      });
      llvm::errs() << '\n';
    }
  }

public:
  using Base::Base;
};
} // namespace
