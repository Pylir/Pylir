//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/InliningUtils.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Util/InlinerUtil.hpp>

#include <memory>

#include "Passes.hpp"

namespace pylir::test {
#define GEN_PASS_DEF_TESTINLINERINTERFACEPASS
#include "Passes.h.inc"
} // namespace pylir::test

namespace {

class TestInlinerInterface
    : public pylir::test::impl::TestInlinerInterfacePassBase<
          TestInlinerInterface> {
protected:
  void runOnOperation() override {
    llvm::SmallVector<mlir::CallOpInterface> calls;
    getOperation()->walk(
        [&](mlir::CallOpInterface call) { calls.push_back(call); });
    mlir::SymbolTableCollection collection;
    for (auto iter : calls) {
      auto ref = mlir::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(
          mlir::dyn_cast<mlir::SymbolRefAttr>(iter.getCallableForCallee()));
      if (!ref || !ref.getValue().starts_with("inline"))
        continue;
      auto func = mlir::dyn_cast_or_null<mlir::CallableOpInterface>(
          iter.resolveCallable(&collection));
      if (!func) {
        iter->emitError("Could not resolve function") << ref;
        signalPassFailure();
        return;
      }
      pylir::Py::inlineCall(iter, func);
    }
  }

public:
  using Base::Base;
};

} // namespace
