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
#include <pylir/Optimizer/PylirPy/IR/PyBuilder.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

using namespace mlir;
using namespace pylir;
using namespace pylir::HIR;
using namespace pylir::Py;
using namespace pylir::Builtins;

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
// Call Conversion Patterns
//===----------------------------------------------------------------------===//

struct CallOpConversionPattern : OpRewritePattern<HIR::CallOp> {
  using OpRewritePattern<HIR::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HIR::CallOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<Py::IterArg> iterArgs;
    SmallVector<Py::DictArg> dictArgs;
    for (const CallArgument& argument : CallArgumentRange(op)) {
      pylir::match(
          argument.kind,
          [&](CallArgument::PositionalTag) {
            iterArgs.emplace_back(argument.value);
          },
          [&](CallArgument::PosExpansionTag) {
            iterArgs.emplace_back(Py::IterExpansion{argument.value});
          },
          [&](CallArgument::MapExpansionTag) {
            dictArgs.emplace_back(Py::MappingExpansion{argument.value});
          },
          [&](StringAttr keyword) {
            Value key = rewriter.create<Py::ConstantOp>(
                op.getLoc(), rewriter.getAttr<Py::StrAttr>(keyword));
            Value hash = rewriter.create<Py::StrHashOp>(op.getLoc(), key);
            dictArgs.emplace_back(Py::DictEntry{key, hash, argument.value});
          });
    }

    Value tuple = rewriter.create<Py::MakeTupleOp>(op.getLoc(), iterArgs);
    Value dict =
        dictArgs.empty()
            ? (Value)rewriter.create<Py::ConstantOp>(
                  op.getLoc(), rewriter.getAttr<Py::DictAttr>())
            : (Value)rewriter.create<Py::MakeDictOp>(op.getLoc(), dictArgs);
    rewriter.replaceOpWithNewOp<Py::CallOp>(
        op, op.getType(), "pylir__call__",
        ValueRange{op.getCallable(), tuple, dict});
    return success();
  }
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
      Value index = builder.create<arith::ConstantIndexOp>(
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

//===----------------------------------------------------------------------===//
// Compiler helper functions
//===----------------------------------------------------------------------===//

Value buildException(PyBuilder& builder, std::string_view kind,
                     std::vector<Py::IterArg> args, Block* exceptionHandler) {
  auto typeObj = builder.createConstant(
      Py::GlobalValueAttr::get(builder.getContext(), kind));
  args.emplace(args.begin(), typeObj);
  Value tuple = builder.createMakeTuple(args, exceptionHandler);
  auto dict = builder.createConstant(builder.getDictAttr());
  auto mro = builder.createTypeMRO(typeObj);
  auto newMethod =
      builder.createMROLookup(mro, Builtins::TypeSlots::New).getResult();

  auto obj = builder.createFunctionCall(newMethod, {newMethod, tuple, dict});
  auto context = builder.createNoneRef();
  builder.createSetSlot(obj, Builtins::BaseExceptionSlots::Context, context);
  auto cause = builder.createNoneRef();
  builder.createSetSlot(obj, Builtins::BaseExceptionSlots::Cause, cause);
  return obj;
}

void implementBlock(OpBuilder& builder, Block* block) {
  PYLIR_ASSERT(block);
  if (auto* next = builder.getBlock()->getNextNode())
    block->insertBefore(next);
  else
    builder.getBlock()->getParent()->push_back(block);

  builder.setInsertionPointToStart(block);
}

Value buildTrySpecialMethodCall(PyBuilder& builder, TypeSlots method,
                                Value args, Value kws, Block* notFoundPath,
                                Block* callIntrException = nullptr) {
  auto element = builder.createTupleGetItem(
      args, builder.create<arith::ConstantIndexOp>(0));
  auto elementType = builder.createTypeOf(element);
  auto mroTuple = builder.createTypeMRO(elementType);
  auto lookup = builder.createMROLookup(mroTuple, method);
  auto failure = builder.createIsUnboundValue(lookup);
  auto* exec = new Block;
  builder.create<cf::CondBranchOp>(builder.getCurrentLoc(), failure,
                                   notFoundPath, exec);

  implementBlock(builder, exec);
  mroTuple = builder.createTypeMRO(builder.createTypeOf(lookup.getResult()));
  auto getMethod = builder.createMROLookup(mroTuple, TypeSlots::Get);
  failure = builder.createIsUnboundValue(getMethod);
  auto* isDescriptor = new Block;
  auto* mergeBlock = new Block;
  mergeBlock->addArgument(builder.getDynamicType(), builder.getCurrentLoc());
  builder.create<cf::CondBranchOp>(builder.getCurrentLoc(), failure, mergeBlock,
                                   ValueRange{lookup.getResult()}, isDescriptor,
                                   ValueRange{});

  implementBlock(builder, isDescriptor);
  auto tuple = builder.createMakeTuple({element, elementType}, nullptr);
  auto result = builder.createPylirCallIntrinsic(getMethod.getResult(), tuple,
                                                 builder.createMakeDict(),
                                                 callIntrException);
  builder.create<cf::BranchOp>(builder.getCurrentLoc(), mergeBlock, result);

  implementBlock(builder, mergeBlock);
  // TODO: This is incorrect. One should be passing all but args[0], as args[0]
  // will already be bound by the __get__ descriptor of function. We haven't yet
  // implemented this however, hence this is the stop gap solution.
  return builder.createPylirCallIntrinsic(mergeBlock->getArgument(0), args, kws,
                                          callIntrException);
}

Value buildSpecialMethodCall(PyBuilder& builder, TypeSlots method, Value args,
                             Value kws, Block* callIntrException = nullptr) {
  auto* notFound = new Block;
  auto result = buildTrySpecialMethodCall(builder, method, args, kws, notFound,
                                          callIntrException);
  OpBuilder::InsertionGuard guard{builder};
  implementBlock(builder, notFound);
  auto exception = buildException(builder, TypeError.name, {}, nullptr);
  builder.createRaise(exception);
  return result;
}

void buildCallOpCompilerBuiltin(PyBuilder& builder,
                                llvm::StringRef functionName) {
  auto func = builder.create<pylir::Py::FuncOp>(
      functionName, builder.getFunctionType({builder.getDynamicType(),
                                             builder.getDynamicType(),
                                             builder.getDynamicType()},
                                            builder.getDynamicType()));
  OpBuilder::InsertionGuard guard{builder};
  builder.setInsertionPointToStart(func.addEntryBlock());

  auto self = func.getArgument(0);
  auto args = func.getArgument(1);
  auto kws = func.getArgument(2);

  auto selfType = builder.createTypeOf(self);
  // We have to somehow break this recursion by detecting a function type and
  // calling it directly.
  auto isFunction = builder.createIs(selfType, builder.createFunctionRef());
  auto* isFunctionBlock = new Block;
  auto* notFunctionBlock = new Block;
  builder.create<cf::CondBranchOp>(isFunction, isFunctionBlock,
                                   notFunctionBlock);

  implementBlock(builder, isFunctionBlock);
  Value result = builder.createFunctionCall(self, {self, args, kws});
  builder.create<Py::ReturnOp>(result);

  implementBlock(builder, notFunctionBlock);
  result = buildSpecialMethodCall(builder, TypeSlots::Call,
                                  builder.createTuplePrepend(self, args), kws);
  builder.create<Py::ReturnOp>(result);
}

} // namespace

void ConvertPylirHIRToPylirPy::runOnOperation() {
  PyBuilder builder(&getContext());
  builder.setInsertionPointToEnd(getOperation().getBody());
  buildCallOpCompilerBuiltin(builder, "pylir__call__");

  ConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](auto...) { return true; });

  target.addIllegalDialect<HIR::PylirHIRDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<InitOpConversionPattern, ReturnOpLowering<InitReturnOp>,
               ReturnOpLowering<HIR::ReturnOp>, GlobalFuncOpConversionPattern,
               CallOpConversionPattern>(&getContext());
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}
