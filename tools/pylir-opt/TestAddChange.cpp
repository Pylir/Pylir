//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include "Passes.hpp"
#include "TestDialect.hpp"

namespace pylir::test {
#define GEN_PASS_DEF_TESTADDCHANGEPASS
#include "Passes.h.inc"
} // namespace pylir::test

namespace {
class TestAddChangePass
    : public pylir::test::impl::TestAddChangePassBase<TestAddChangePass> {
protected:
  void runOnOperation() override {
    auto builder = mlir::OpBuilder::atBlockEnd(getOperation().getBody());
    builder.create<pylir::test::ChangeOp>(builder.getUnknownLoc());
  }

public:
  using Base::Base;
};
} // namespace
