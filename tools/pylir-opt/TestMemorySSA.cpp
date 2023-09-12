//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/Analysis/MemorySSA.hpp>

#include <memory>

#include "Passes.hpp"

namespace pylir::test {
#define GEN_PASS_DEF_TESTMEMORYSSAPASS
#include "Passes.h.inc"
} // namespace pylir::test

namespace {

class TestMemorySSA
    : public pylir::test::impl::TestMemorySSAPassBase<TestMemorySSA> {
protected:
  void runOnOperation() override;

public:
  using Base::Base;
};

void TestMemorySSA::runOnOperation() {
  for (auto func : getOperation().getOps<mlir::FunctionOpInterface>()) {
    llvm::outs() << getChildAnalysis<pylir::MemorySSA>(func);
  }
}
} // namespace
