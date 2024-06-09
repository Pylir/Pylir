//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Interfaces/Builtins.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

#include "Passes.hpp"

namespace pylir::Py {
#define GEN_PASS_DEF_EXPANDPYDIALECTPASS
#include "pylir/Optimizer/PylirPy/Transforms/Passes.h.inc"
} // namespace pylir::Py

namespace {

mlir::Value callOrInvoke(
    mlir::Location loc, mlir::OpBuilder& builder, mlir::Block* dest,
    const pylir::Builtins::Builtin& callable,
    llvm::ArrayRef<pylir::Py::IterArg> args,
    llvm::PointerUnion<pylir::Py::ExceptionHandlingInterface, mlir::Block*>
        exceptionHandler) {
  mlir::Value result;
  if (exceptionHandler) {
    auto* happyPath = new mlir::Block;
    mlir::Block* exceptionPath;
    mlir::ValueRange unwindOperands;
    if (auto interface = mlir::dyn_cast<pylir::Py::ExceptionHandlingInterface>(
            exceptionHandler)) {
      exceptionPath = interface.getExceptionPath();
      unwindOperands = static_cast<mlir::OperandRange>(
          interface.getUnwindDestOperandsMutable());
    } else {
      exceptionPath = exceptionHandler.get<mlir::Block*>();
    }

    result =
        builder
            .create<pylir::Py::InvokeOp>(
                loc, builder.getType<pylir::Py::DynamicType>(),
                pylir::Builtins::PylirCall.name,
                mlir::ValueRange{
                    builder.create<pylir::Py::ConstantOp>(
                        loc, pylir::Py::GlobalValueAttr::get(
                                 builder.getContext(), callable.name)),
                    builder.create<pylir::Py::MakeTupleOp>(loc, args),
                    builder.create<pylir::Py::ConstantOp>(
                        loc, pylir::Py::DictAttr::get(builder.getContext())),
                },
                mlir::ValueRange{}, unwindOperands, happyPath, exceptionPath)
            .getResult(0);
    happyPath->insertBefore(dest);
    builder.setInsertionPointToStart(happyPath);
  } else {
    result =
        builder
            .create<pylir::Py::CallOp>(
                loc, builder.getType<pylir::Py::DynamicType>(),
                pylir::Builtins::PylirCall.name,
                mlir::ValueRange{
                    builder.create<pylir::Py::ConstantOp>(
                        loc, pylir::Py::GlobalValueAttr::get(
                                 builder.getContext(), callable.name)),
                    builder.create<pylir::Py::MakeTupleOp>(loc, args),
                    builder.create<pylir::Py::ConstantOp>(
                        loc, pylir::Py::DictAttr::get(builder.getContext())),
                })
            .getResult(0);
  }
  return result;
}

struct TupleUnrollPattern : mlir::OpRewritePattern<pylir::Py::MakeTupleOp> {
  using mlir::OpRewritePattern<pylir::Py::MakeTupleOp>::OpRewritePattern;

  void rewrite(pylir::Py::MakeTupleOp op,
               mlir::PatternRewriter& rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto list = rewriter.create<pylir::Py::MakeListOp>(
        op.getLoc(), op.getArguments(), op.getIterExpansionAttr());
    rewriter.replaceOpWithNewOp<pylir::Py::ListToTupleOp>(op, list);
  }

  mlir::LogicalResult match(pylir::Py::MakeTupleOp op) const override {
    return mlir::success(!op.getIterExpansion().empty());
  }
};

struct TupleExUnrollPattern : mlir::OpRewritePattern<pylir::Py::MakeTupleExOp> {
  using mlir::OpRewritePattern<pylir::Py::MakeTupleExOp>::OpRewritePattern;

  void rewrite(pylir::Py::MakeTupleExOp op,
               mlir::PatternRewriter& rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto list = rewriter.create<pylir::Py::MakeListExOp>(
        op.getLoc(), op.getArguments(), op.getIterExpansionAttr(),
        op.getNormalDestOperands(), op.getUnwindDestOperands(),
        op.getHappyPath(), op.getExceptionPath());
    rewriter.setInsertionPointToStart(list.getHappyPath());
    rewriter.replaceOpWithNewOp<pylir::Py::ListToTupleOp>(op, list);
  }

  mlir::LogicalResult match(pylir::Py::MakeTupleExOp op) const override {
    return mlir::success(!op.getIterExpansion().empty());
  }
};

/// Turns exception handling variants into non-exception handling variants if
/// they do not have any expansions.
struct ExRemovePattern
    : mlir::OpInterfaceRewritePattern<pylir::Py::ExceptionHandlingInterface> {
  using mlir::OpInterfaceRewritePattern<
      pylir::Py::ExceptionHandlingInterface>::OpInterfaceRewritePattern;

  void rewrite(pylir::Py::ExceptionHandlingInterface op,
               mlir::PatternRewriter& rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Operation* clone = op.cloneWithoutExceptionHandling(rewriter);
    rewriter.create<mlir::cf::BranchOp>(loc, op.getHappyPath(),
                                        op.getNormalDestOperands());
    rewriter.replaceOp(op, clone);
  }

  mlir::LogicalResult
  match(pylir::Py::ExceptionHandlingInterface op) const override {
    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
        .Case([](pylir::Py::MakeDictExOp op) {
          return mlir::success(op.getMappingExpansionAttr().empty());
        })
        .Case<pylir::Py::MakeTupleExOp, pylir::Py::MakeListExOp,
              pylir::Py::MakeSetExOp>([](auto op) {
          return mlir::success(op.getIterExpansion().empty());
        })
        .Default(mlir::failure());
  }
};

void raiseException(mlir::Value exception,
                    pylir::Py::ExceptionHandlingInterface exceptionHandler,
                    mlir::PatternRewriter& rewriter, mlir::Location loc) {
  if (!exceptionHandler) {
    rewriter.create<pylir::Py::RaiseOp>(loc, exception);
  } else {
    llvm::SmallVector<mlir::Value> args = {exception};
    mlir::OperandRange unwindArgs =
        exceptionHandler.getUnwindDestOperandsMutable();
    args.append(unwindArgs.begin(), unwindArgs.end());
    rewriter.create<mlir::cf::BranchOp>(
        loc, exceptionHandler.getExceptionPath(), args);
  }
}

void restIterIntoList(mlir::Block* dest, mlir::Location loc,
                      mlir::Value iterObject, mlir::Block* conditionBlock,
                      mlir::PatternRewriter& rewriter, mlir::Value list,
                      pylir::Py::ExceptionHandlingInterface exceptionHandler) {
  conditionBlock->insertBefore(dest);
  rewriter.setInsertionPointToStart(conditionBlock);
  auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto* stopIterationHandler = new mlir::Block;
  stopIterationHandler->addArgument(rewriter.getType<pylir::Py::DynamicType>(),
                                    loc);

  auto next = callOrInvoke(loc, rewriter, dest, pylir::Builtins::Next,
                           {iterObject}, stopIterationHandler);

  auto len = rewriter.create<pylir::Py::ListLenOp>(loc, list);
  auto newLen = rewriter.create<mlir::arith::AddIOp>(loc, len, one);
  rewriter.create<pylir::Py::ListResizeOp>(loc, list, newLen);
  rewriter.create<pylir::Py::ListSetItemOp>(loc, list, len, next);
  rewriter.create<mlir::cf::BranchOp>(loc, conditionBlock);

  stopIterationHandler->insertBefore(dest);
  rewriter.setInsertionPointToStart(stopIterationHandler);
  auto stopIterationType = rewriter.create<pylir::Py::ConstantOp>(
      loc, pylir::Py::GlobalValueAttr::get(
               loc.getContext(), pylir::Builtins::StopIteration.name));
  auto typeOf = rewriter.create<pylir::Py::TypeOfOp>(
      loc, stopIterationHandler->getArgument(0));
  auto isStopIteration =
      rewriter.create<pylir::Py::IsOp>(loc, stopIterationType, typeOf);
  auto* continueBlock = new mlir::Block;
  auto* reraiseBlock = new mlir::Block;
  rewriter.create<mlir::cf::CondBranchOp>(loc, isStopIteration, continueBlock,
                                          reraiseBlock);

  reraiseBlock->insertBefore(dest);
  rewriter.setInsertionPointToStart(reraiseBlock);
  raiseException(stopIterationHandler->getArgument(0), exceptionHandler,
                 rewriter, loc);

  continueBlock->insertBefore(dest);
  rewriter.setInsertionPointToStart(continueBlock);
}

void continueOntoHappyPath(pylir::Py::ExceptionHandlingInterface interface,
                           mlir::PatternRewriter& rewriter) {
  if (!interface)
    return;

  rewriter.setInsertionPointAfter(interface);
  mlir::Block* happyPath = interface.getHappyPath();
  if (!happyPath->getSinglePredecessor()) {
    rewriter.template create<mlir::cf::BranchOp>(interface.getLoc(), happyPath);
  } else {
    // interface is the single predecessor, and dropping its successor is really
    // an update on 'interface', not the block.
    rewriter.modifyOpInPlace(interface, [=] { happyPath->dropAllUses(); });
    rewriter.mergeBlocks(happyPath, interface->getBlock(),
                         static_cast<mlir::OperandRange>(
                             interface.getNormalDestOperandsMutable()));
  }
}

template <class TargetOp>
struct ListUnrollPattern : mlir::OpRewritePattern<TargetOp> {
  using mlir::OpRewritePattern<TargetOp>::OpRewritePattern;

  void rewrite(TargetOp op, mlir::PatternRewriter& rewriter) const override {
    auto exceptionHandler =
        mlir::dyn_cast<pylir::Py::ExceptionHandlingInterface>(*op);

    auto block = op->getBlock();
    auto dest = block->splitBlock(op);
    rewriter.setInsertionPointToEnd(block);
    auto loc = op.getLoc();
    auto range = op.getIterExpansion();
    PYLIR_ASSERT(!range.empty());
    auto begin = range.begin();
    auto prefix = op.getOperands().take_front(*begin);
    auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto list = rewriter.create<pylir::Py::MakeListOp>(
        loc, prefix, rewriter.getDenseI32ArrayAttr({}));
    for (const auto& iter :
         llvm::drop_begin(llvm::enumerate(op.getOperands()), *begin)) {
      if (begin == range.end() ||
          static_cast<std::size_t>(*begin) != iter.index()) {
        auto len = rewriter.create<pylir::Py::ListLenOp>(loc, list);
        auto newLen = rewriter.create<mlir::arith::AddIOp>(loc, len, one);
        rewriter.create<pylir::Py::ListResizeOp>(loc, list, newLen);
        rewriter.create<pylir::Py::ListSetItemOp>(loc, list, len, iter.value());
        continue;
      }
      begin++;
      auto iterObject = callOrInvoke(loc, rewriter, dest, pylir::Builtins::Iter,
                                     {iter.value()}, exceptionHandler);
      auto* condition = new mlir::Block;
      rewriter.create<mlir::cf::BranchOp>(loc, condition);

      restIterIntoList(dest, loc, iterObject, condition, rewriter, list,
                       exceptionHandler);
    }
    rewriter.mergeBlocks(dest, rewriter.getBlock());

    continueOntoHappyPath(exceptionHandler, rewriter);
    rewriter.replaceOp(op, list);
  }

  mlir::LogicalResult match(TargetOp op) const override {
    return mlir::success(!op.getIterExpansion().empty());
  }
};

template <class TargetOp>
struct UnpackOpPattern : mlir::OpRewritePattern<TargetOp> {
  using mlir::OpRewritePattern<TargetOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TargetOp op, mlir::PatternRewriter& rewriter) const override {
    llvm::SmallVector<mlir::Value> replacementValues;

    auto exceptionHandler =
        mlir::dyn_cast<pylir::Py::ExceptionHandlingInterface>(*op);

    mlir::Block* block = op->getBlock();
    auto* dest = block->splitBlock(op);
    rewriter.setInsertionPointToEnd(block);
    mlir::Location loc = op.getLoc();
    auto stopIterationType = rewriter.create<pylir::Py::ConstantOp>(
        loc, pylir::Py::GlobalValueAttr::get(
                 this->getContext(), pylir::Builtins::StopIteration.name));

    auto iterObject = callOrInvoke(loc, rewriter, dest, pylir::Builtins::Iter,
                                   {op.getIterable()}, exceptionHandler);

    auto* valueError = new mlir::Block;
    valueError->addArgument(rewriter.getType<pylir::Py::DynamicType>(), loc);
    for (mlir::Value iter : op.getBefore()) {
      (void)iter;
      auto next = callOrInvoke(loc, rewriter, dest, pylir::Builtins::Next,
                               {iterObject}, valueError);
      replacementValues.push_back(next);
    }

    auto* afterBeforeBlock = rewriter.getBlock();

    valueError->insertBefore(dest);
    rewriter.setInsertionPointToStart(valueError);
    // If the iterator is exhausted we need to raise a ValueError, not a
    // StopIteration exception. Check if it's StopIteration, to raise a
    // ValueError, if not reraise.
    auto exc = valueError->getArgument(0);
    auto exceptionType = rewriter.create<pylir::Py::TypeOfOp>(loc, exc);
    auto isStopIteration =
        rewriter.create<pylir::Py::IsOp>(loc, exceptionType, stopIterationType);
    auto* reraiseBlock = new mlir::Block;
    auto* valueErrorBlock = new mlir::Block;
    rewriter.create<mlir::cf::CondBranchOp>(loc, isStopIteration,
                                            valueErrorBlock, reraiseBlock);

    reraiseBlock->insertBefore(dest);
    rewriter.setInsertionPointToStart(reraiseBlock);
    raiseException(exc, exceptionHandler, rewriter, loc);

    valueErrorBlock->insertBefore(dest);
    rewriter.setInsertionPointToStart(valueErrorBlock);
    auto valueErrorExc = callOrInvoke(
        loc, rewriter, dest, pylir::Builtins::ValueError, {}, exceptionHandler);
    raiseException(valueErrorExc, exceptionHandler, rewriter, loc);

    rewriter.setInsertionPointToEnd(afterBeforeBlock);
    if (!op.getRest()) {
      // There are no rest args. We have to raise a ValueError if the iterator
      // is not exhausted. In other words, if it has more elements than we
      // unpacked into.
      auto* emptyIterCheck = new mlir::Block;
      emptyIterCheck->addArgument(rewriter.getType<pylir::Py::DynamicType>(),
                                  loc);
      callOrInvoke(loc, rewriter, dest, pylir::Builtins::Next, {iterObject},
                   emptyIterCheck);
      // If it was not exhausted we are continuing from the above call. Branch
      // to the valueErrorBlock to raise the value error.
      rewriter.create<mlir::cf::BranchOp>(loc, valueErrorBlock);

      emptyIterCheck->insertBefore(dest);
      rewriter.setInsertionPointToStart(emptyIterCheck);
      exc = emptyIterCheck->getArgument(0);
      exceptionType = rewriter.create<pylir::Py::TypeOfOp>(loc, exc);
      isStopIteration = rewriter.create<pylir::Py::IsOp>(loc, exceptionType,
                                                         stopIterationType);
      reraiseBlock = new mlir::Block;
      rewriter.create<mlir::cf::CondBranchOp>(loc, isStopIteration, dest,
                                              reraiseBlock);

      reraiseBlock->insertBefore(dest);
      rewriter.setInsertionPointToStart(reraiseBlock);
      raiseException(exc, exceptionHandler, rewriter, loc);

      continueOntoHappyPath(exceptionHandler, rewriter);
      rewriter.replaceOp(op, replacementValues);
      return mlir::success();
    }

    auto list = rewriter.create<pylir::Py::MakeListOp>(loc);
    replacementValues.push_back(list);
    auto* conditionBlock = new mlir::Block;
    rewriter.create<mlir::cf::BranchOp>(loc, conditionBlock);

    // All elements following the before elements get inserted into the list. We
    // then pop the after elements from the list.
    restIterIntoList(dest, loc, iterObject, conditionBlock, rewriter, list,
                     exceptionHandler);

    auto listSize = rewriter.create<pylir::Py::ListLenOp>(loc, list);
    auto argAfterSize = rewriter.create<mlir::arith::ConstantIndexOp>(
        loc, op.getAfter().size());
    auto tooFew = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, listSize, argAfterSize);
    rewriter.create<mlir::cf::CondBranchOp>(loc, tooFew, valueErrorBlock, dest);

    rewriter.setInsertionPointToStart(dest);
    for (std::size_t index : llvm::reverse(
             llvm::seq_inclusive<std::size_t>(1, op.getAfter().size()))) {
      auto calcIndex = rewriter.create<mlir::arith::SubIOp>(
          loc, listSize,
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, index));
      replacementValues.emplace_back(
          rewriter.create<pylir::Py::ListGetItemOp>(loc, list, calcIndex));
    }
    auto newSize =
        rewriter.create<mlir::arith::SubIOp>(loc, listSize, argAfterSize);
    rewriter.create<pylir::Py::ListResizeOp>(loc, list, newSize);

    continueOntoHappyPath(exceptionHandler, rewriter);
    rewriter.replaceOp(op, replacementValues);
    return mlir::success();
  }
};

struct ExpandPyDialectPass
    : public pylir::Py::impl::ExpandPyDialectPassBase<ExpandPyDialectPass> {
  using Base::Base;

protected:
  void runOnOperation() override;
};

void ExpandPyDialectPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addDynamicallyLegalOp<pylir::Py::MakeTupleOp, pylir::Py::MakeListOp,
                               pylir::Py::MakeSetOp, pylir::Py::MakeDictOp>(
      [](mlir::Operation* op) -> bool {
        return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case([](pylir::Py::MakeDictOp op) {
              return op.getMappingExpansionAttr().empty();
            })
            .Case<pylir::Py::MakeTupleOp, pylir::Py::MakeListOp,
                  pylir::Py::MakeSetOp>(
                [](auto op) { return op.getIterExpansion().empty(); })
            .Default(false);
      });
  target.addIllegalOp<pylir::Py::MakeTupleExOp, pylir::Py::MakeListExOp,
                      pylir::Py::MakeSetExOp, pylir::Py::MakeDictExOp,
                      pylir::Py::UnpackOp, pylir::Py::UnpackExOp>();
  target.markUnknownOpDynamicallyLegal([](auto...) { return true; });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<TupleUnrollPattern>(&getContext());
  patterns.add<TupleExUnrollPattern>(&getContext());
  patterns.add<ListUnrollPattern<pylir::Py::MakeListOp>>(&getContext());
  patterns.add<ListUnrollPattern<pylir::Py::MakeListExOp>>(&getContext());
  patterns.add<UnpackOpPattern<pylir::Py::UnpackOp>>(&getContext());
  patterns.add<UnpackOpPattern<pylir::Py::UnpackExOp>>(&getContext());
  patterns.add<ExRemovePattern>(&getContext());
  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}
} // namespace
