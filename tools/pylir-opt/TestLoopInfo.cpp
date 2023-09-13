//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/FunctionInterfaces.h>

#include <llvm/ADT/PostOrderIterator.h>

#include <pylir/Optimizer/Analysis/LoopInfo.hpp>

#include "Passes.hpp"

namespace pylir::test {
#define GEN_PASS_DEF_TESTLOOPINFOPASS
#include "Passes.h.inc"
} // namespace pylir::test

namespace {
class TestLoopInfo
    : public pylir::test::impl::TestLoopInfoPassBase<TestLoopInfo> {
public:
  using Base::Base;

protected:
  void runOnOperation() override {
    for (auto iter : getOperation().getOps<mlir::FunctionOpInterface>()) {
      getChildAnalysis<pylir::LoopInfo>(iter).print(llvm::outs());
    }
  }
};
} // namespace
