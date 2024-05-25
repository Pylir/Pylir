//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseMap.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemOps.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>
#include <pylir/Support/Variant.hpp>

namespace pylir {
#define GEN_PASS_DEF_CONVERTPYLIRPYTOPYLIRMEMPASS
#include "pylir/Optimizer/Conversion/Passes.h.inc"
} // namespace pylir

namespace {

struct MakeDictOpConversion : mlir::OpRewritePattern<pylir::Py::MakeDictOp> {
  using mlir::OpRewritePattern<pylir::Py::MakeDictOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pylir::Py::MakeDictOp op,
                  mlir::PatternRewriter& rewriter) const override {
    auto dict = rewriter.create<pylir::Py::ConstantOp>(
        op.getLoc(), pylir::Py::GlobalValueAttr::get(
                         getContext(), pylir::Builtins::Dict.name));
    auto slotCount = rewriter.create<pylir::Py::TupleLenOp>(
        op.getLoc(),
        rewriter.create<pylir::Py::TypeSlotsOp>(op.getLoc(), dict));
    auto mem = rewriter.create<pylir::Mem::GCAllocObjectOp>(op.getLoc(), dict,
                                                            slotCount);
    auto init = rewriter.replaceOpWithNewOp<pylir::Mem::InitDictOp>(
        op, op.getType(), mem);
    for (auto arg : op.getDictArgs()) {
      auto& entry = pylir::get<pylir::Py::DictEntry>(arg);
      rewriter.create<pylir::Py::DictSetItemOp>(op.getLoc(), init, entry.key,
                                                entry.hash, entry.value);
    }
    return mlir::success();
  }
};

struct ConvertPylirPyToPylirMem
    : pylir::impl::ConvertPylirPyToPylirMemPassBase<ConvertPylirPyToPylirMem> {
protected:
  void runOnOperation() override;

public:
  using Base::Base;
};

#include "pylir/Optimizer/Conversion/PylirPyToPylirMem/PylirPyToPylirMem.cpp.inc"

void ConvertPylirPyToPylirMem::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](auto...) { return true; });

  target.addIllegalOp<
      pylir::Py::MakeTupleOp, pylir::Py::MakeListOp, pylir::Py::MakeSetOp,
      pylir::Py::MakeDictOp, pylir::Py::MakeFuncOp, pylir::Py::MakeObjectOp,
      pylir::Py::ListToTupleOp, pylir::Py::BoolFromI1Op,
      pylir::Py::IntFromSignedOp, pylir::Py::IntFromUnsignedOp,
      pylir::Py::StrConcatOp, pylir::Py::IntToStrOp, pylir::Py::StrCopyOp,
      pylir::Py::TupleDropFrontOp, pylir::Py::TuplePrependOp,
      pylir::Py::IntAddOp, pylir::Py::TupleCopyOp, pylir::Py::FloatFromF64,
      pylir::Py::MakeTypeOp>();

  mlir::RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  patterns.insert<MakeDictOpConversion>(&getContext());
  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}
} // namespace
