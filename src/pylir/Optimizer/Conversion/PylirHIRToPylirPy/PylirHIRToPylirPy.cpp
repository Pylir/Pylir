//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/ScopeExit.h>

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
// Function Conversion Patterns
//===----------------------------------------------------------------------===//

/// Creates a new 'py.func' to translate from the universal calling convention
/// to the parameters of 'implementation'. 'builder' will be used to create any
/// MLIR operations. 'calleeSymbol' should refer to the symbol corresponding to
/// 'implementation' after dialect conversion.
Py::FuncOp buildFunctionCC(OpBuilder& builder, GlobalFuncOp implementation,
                           FlatSymbolRefAttr calleeSymbol) {
  Location loc = implementation.getLoc();
  auto dynamicType = builder.getType<DynamicType>();
  auto cc = builder.create<Py::FuncOp>(
      loc, implementation.getName(),
      FunctionType::get(builder.getContext(),
                        {dynamicType, dynamicType, dynamicType},
                        {dynamicType}));
  OpBuilder::InsertionGuard guard{builder};
  builder.setInsertionPointToStart(cc.addEntryBlock());

  Value closure = cc.getArgument(0);
  Value tuple = cc.getArgument(1);
  Value dict = cc.getArgument(2);

  Value defaultTuple = builder.create<Py::GetSlotOp>(
      loc, closure,
      builder.create<arith::ConstantIndexOp>(
          loc, static_cast<std::size_t>(Builtins::FunctionSlots::Defaults)));
  Value kwDefaultDict = builder.create<Py::GetSlotOp>(
      loc, closure,
      builder.create<arith::ConstantIndexOp>(
          loc, static_cast<std::size_t>(Builtins::FunctionSlots::KwDefaults)));

  Value tupleLen = builder.create<Py::TupleLenOp>(loc, tuple);

  Value unboundValue =
      builder.create<Py::ConstantOp>(loc, builder.getAttr<Py::UnboundAttr>());
  std::size_t positionalArgsSeen = 0;
  std::size_t positionalDefaultArgsSeen = 0;
  std::optional<std::size_t> positionalRestArgsPos;
  std::optional<std::size_t> kwRestArgsPos;
  SmallVector<Value> callArguments;
  for (HIR::FunctionParameter parameter :
       HIR::FunctionParameterRange(implementation)) {
    // There can only be one rest-parameter of each kind. These will be set
    // at the end after all other parameters are converted. The index in the
    // call parameter array with a null-placeholder are set here already.
    if (parameter.isKeywordRest()) {
      kwRestArgsPos = callArguments.size();
      callArguments.emplace_back();
      continue;
    }
    if (parameter.isPosRest()) {
      positionalRestArgsPos = callArguments.size();
      callArguments.emplace_back();
      continue;
    }

    // Current value of the argument that will be placed in the arguments array
    // at the end of the loop body.
    Value currentArg = unboundValue;
    auto atExit =
        llvm::make_scope_exit([&] { callArguments.emplace_back(currentArg); });

    if (!parameter.isKeywordOnly()) {
      // Checks whether a positional argument for the parameter is present in
      // the tuple.
      Value index =
          builder.create<arith::ConstantIndexOp>(loc, positionalArgsSeen++);
      Value inTuple = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, index, tupleLen);

      auto* hasValue = cc.addBlock();
      auto* continueSearch = cc.addBlock();
      currentArg =
          continueSearch->addArgument(builder.getType<Py::DynamicType>(), loc);
      builder.create<cf::CondBranchOp>(loc, inTuple, hasValue, continueSearch,
                                       unboundValue);

      builder.setInsertionPointToStart(hasValue);
      Value value = builder.create<Py::TupleGetItemOp>(loc, tuple, index);
      builder.create<cf::BranchOp>(loc, continueSearch, value);
      builder.setInsertionPointToStart(continueSearch);
    }

    Value keyword;
    Value hash;
    if (!parameter.isPositionalOnly()) {
      // If the parameter is callable using the keyword-syntax, check the
      // dictionary as well.
      keyword = builder.create<Py::ConstantOp>(
          loc, builder.getAttr<Py::StrAttr>(parameter.getName()));
      hash = builder.create<Py::StrHashOp>(loc, keyword);
      Value lookup =
          builder.create<Py::DictTryGetItemOp>(loc, dict, keyword, hash);
      Value failure = builder.create<Py::IsUnboundValueOp>(loc, lookup);

      auto* foundBlock = cc.addBlock();
      auto* continueBlock = cc.addBlock();
      continueBlock->addArgument(currentArg.getType(), loc);
      builder.create<cf::CondBranchOp>(loc, failure, continueBlock, currentArg,
                                       foundBlock, ValueRange{});

      builder.setInsertionPointToStart(foundBlock);
      // Delete the entry from the argument dictionary for the rest parameter.
      builder.create<Py::DictDelItemOp>(loc, dict, keyword, hash);

      // It is an error for a parameter to be bound twice (once through
      // positional argument, again through keyword argument).
      Value notFoundPreviously =
          builder.create<Py::IsUnboundValueOp>(loc, currentArg);
      // TODO: This should raise a 'TypeError'.
      builder.create<cf::AssertOp>(
          loc, notFoundPreviously,
          "keyword arg matched previous positional arg");
      builder.create<cf::BranchOp>(loc, continueBlock, lookup);

      builder.setInsertionPointToStart(continueBlock);
      currentArg = continueBlock->getArgument(0);
    }

    // Default parameter handling.
    Value notFound = builder.create<Py::IsUnboundValueOp>(loc, currentArg);
    if (!parameter.hasDefault()) {
      // TODO: This should raise a 'TypeError'.
      builder.create<cf::AssertOp>(loc, notFound,
                                   "failed to find argument for parameter");
      continue;
    }

    // Depending on whether the parameter is a keyword-only parameter or not,
    // the default value is either read from the default tuple or the keyword
    // defaults dictionary.
    Block* needsDefault = cc.addBlock();
    Block* afterDefault = cc.addBlock();
    afterDefault->addArgument(currentArg.getType(), loc);
    builder.create<cf::CondBranchOp>(loc, notFound, needsDefault, afterDefault,
                                     currentArg);

    builder.setInsertionPointToStart(needsDefault);
    if (parameter.isKeywordOnly()) {
      Value lookup = builder.create<Py::DictTryGetItemOp>(loc, kwDefaultDict,
                                                          keyword, hash);
      builder.create<cf::BranchOp>(loc, afterDefault, lookup);
    } else {
      Value index = builder.create<mlir::arith::ConstantIndexOp>(
          loc, positionalDefaultArgsSeen++);
      Value lookup =
          builder.create<Py::TupleGetItemOp>(loc, defaultTuple, index);
      builder.create<cf::BranchOp>(loc, afterDefault, lookup);
    }

    builder.setInsertionPointToStart(afterDefault);
    currentArg = afterDefault->getArgument(0);
    notFound = builder.create<Py::IsUnboundValueOp>(loc, currentArg);
    // TODO: This should raise a 'TypeError'.
    builder.create<cf::AssertOp>(loc, notFound,
                                 "failed to find argument for parameter");
  }

  if (positionalRestArgsPos) {
    callArguments[*positionalRestArgsPos] =
        builder.create<Py::TupleDropFrontOp>(
            loc,
            builder.create<arith::ConstantIndexOp>(loc, positionalArgsSeen),
            tuple);
  }
  if (kwRestArgsPos)
    callArguments[*kwRestArgsPos] = dict;

  Value ret =
      builder.create<Py::CallOp>(loc, dynamicType, calleeSymbol, callArguments)
          .getResult(0);
  builder.create<Py::ReturnOp>(loc, ret);

  return cc;
}

constexpr llvm::StringRef functionImplSuffix = "$impl";

struct GlobalFuncOpConversionPattern : OpRewritePattern<GlobalFuncOp> {
  using OpRewritePattern<GlobalFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalFuncOp op,
                                PatternRewriter& rewriter) const override {
    // Global func splits into two functions:
    // * the implementation function with the suffix "$impl",
    // * the CC function copying the name.
    // The latter always has 'object(function, tuple, dict)' as calling
    // convention.
    auto functionImpl = rewriter.create<Py::FuncOp>(
        op.getLoc(), op.getName() + functionImplSuffix, op.getFunctionType());
    buildFunctionCC(rewriter, op, FlatSymbolRefAttr::get(functionImpl));

    rewriter.inlineRegionBefore(op.getBody(), functionImpl.getBody(),
                                functionImpl.getBody().end());
    functionImpl.setArgAttrsAttr(op.getArgAttrsAttr());
    functionImpl.setResAttrsAttr(op.getResAttrsAttr());

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Module Conversion Patterns
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
  patterns.add<InitOpConversionPattern, ReturnOpLowering<InitReturnOp>,
               ReturnOpLowering<HIR::ReturnOp>, GlobalFuncOpConversionPattern>(
      &getContext());
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}
