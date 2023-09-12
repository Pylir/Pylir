//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pylir/Optimizer/Linker/Linker.hpp>

#include "Passes.hpp"

namespace pylir::test {
#define GEN_PASS_DEF_TESTLINKERPASS
#include "Passes.h.inc"
} // namespace pylir::test

namespace {
class TestLinker : public pylir::test::impl::TestLinkerPassBase<TestLinker> {
protected:
  void runOnOperation() override {
    llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
    llvm::transform(
        llvm::make_early_inc_range(getOperation().getOps<mlir::ModuleOp>()),
        std::back_inserter(modules), [](mlir::ModuleOp moduleOp) {
          moduleOp->remove();
          return moduleOp;
        });
    auto linked = pylir::linkModules(modules);
    getOperation().push_back(linked.release());
  }

public:
  using Base::Base;
};
} // namespace
