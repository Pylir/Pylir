// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>

#include <pylir/Optimizer/PylirHIR/IR/PylirHIROps.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Util/ExceptionRewriter.hpp>

#include "Passes.hpp"

using namespace mlir;
using namespace pylir;
using namespace pylir::Py;

namespace pylir::HIR {
#define GEN_PASS_DEF_CLASSBODYOUTLININGPASS
#include "pylir/Optimizer/PylirHIR/Transforms/Passes.h.inc"
} // namespace pylir::HIR

namespace {
class ClassBodyOutlining
    : public pylir::HIR::impl::ClassBodyOutliningPassBase<ClassBodyOutlining> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};

struct ClassBodyOutliningPattern
    : Py::OpExRewritePattern<HIR::ClassOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(HIR::ClassOpInterface op,
                                ExceptionRewriter& rewriter) const override {
    auto funcOp = rewriter.create<HIR::FuncOp>(
        op.getLoc(), op.getName(), ArrayRef{HIR::FunctionParameterSpec()});
    // Erase entry block created by the builder.
    rewriter.eraseBlock(&funcOp.getBody().front());
    rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(),
                                funcOp.getBody().end());

    rewriter.replaceOpWithNewOp<HIR::BuildClassOp>(
        op, funcOp, op.getName(), op.getArguments(), op.getKeywords(),
        op.getKindInternal());
    return success();
  }
};

struct ClassReturnPattern : OpRewritePattern<HIR::ClassReturnOp> {
  using OpRewritePattern<HIR::ClassReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HIR::ClassReturnOp op,
                                PatternRewriter& rewriter) const override {
    Value none = rewriter.create<Py::ConstantOp>(
        op.getLoc(),
        rewriter.getAttr<Py::GlobalValueAttr>(Builtins::None.name));
    rewriter.replaceOpWithNewOp<HIR::ReturnOp>(op, none);
    return success();
  }
};

} // namespace

void ClassBodyOutlining::runOnOperation() {
  ConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](auto...) { return true; });

  target.addIllegalOp<HIR::ClassOp, HIR::ClassExOp, HIR::ClassReturnOp>();

  RewritePatternSet patterns(&getContext());
  patterns.add<ClassBodyOutliningPattern, ClassReturnPattern>(&getContext());
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}
