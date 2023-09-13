//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>

#include "Passes.hpp"

namespace pylir::test {
#define GEN_PASS_DEF_TESTHELLOWORLDPASS
#include "Passes.h.inc"
} // namespace pylir::test

namespace {
class TestHelloWorld
    : public pylir::test::impl::TestHelloWorldPassBase<TestHelloWorld> {
protected:
  void runOnOperation() override {
    // Using cout here as it has an atomicity guarantee for single invocations
    // of operator<<
    std::cout << "Hello World!\n";
  }

public:
  using Base::Base;
};
} // namespace
