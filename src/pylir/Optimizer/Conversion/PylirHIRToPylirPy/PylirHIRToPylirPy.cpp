//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirHIR/IR/PylirHIRDialect.hpp>
#include <pylir/Optimizer/PylirHIR/IR/PylirHIROps.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

using namespace mlir;
using namespace pylir;
using namespace pylir::HIR;
using namespace pylir::Py;

namespace pylir {
#define GEN_PASS_DEF_CONVERTPYLIRHIRTOPYLIRPYPASS
#include "pylir/Optimizer/Conversion/Passes.h.inc"
} // namespace pylir

namespace {
struct ConvertPylirHIRToPylirPy
    : pylir::impl::ConvertPylirHIRToPylirPyPassBase<ConvertPylirHIRToPylirPy> {
protected:
  void runOnOperation() override;

public:
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

struct InitOpConversionPattern : OpRewritePattern<InitOp> {
  using OpRewritePattern<InitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InitOp op,
                                PatternRewriter& rewriter) const override {
    std::string functionName = "__init__";
    // The main init is treate specially as it is the entry point of the whole
    // python program by default. It is callable with an `__init__` as that is a
    // valid C identifier.
    if (!op.isMainModule())
      functionName = (op.getName() + "." + functionName).str();

    auto funcOp = rewriter.create<Py::FuncOp>(
        op->getLoc(), functionName,
        rewriter.getFunctionType(/*inputs=*/{},
                                 rewriter.getType<DynamicType>()));
    // The region can be inlined directly without creating a suitable entry
    // block for the function as the function body does not need any block
    // arguments.
    rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(),
                                funcOp.getBody().end());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering pattern for any Op that is `ReturnLike` to `py.return`.
/// Returns ALL its operands.
template <class OpT>
struct ReturnOpLowering : OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  static_assert(OpT::template hasTrait<OpTrait::ReturnLike>());

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<Py::ReturnOp>(op, op->getOperands());
    return success();
  }
};

} // namespace

void ConvertPylirHIRToPylirPy::runOnOperation() {
  ConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](auto...) { return true; });

  target.addIllegalDialect<HIR::PylirHIRDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<InitOpConversionPattern, ReturnOpLowering<InitReturnOp>>(
      &getContext());
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}
