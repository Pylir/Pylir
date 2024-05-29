// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/PatternMatch.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyTraits.hpp>

namespace pylir::Py {
/// Custom 'PatternRewriter' performing on-the-fly transformation of rewriters
/// to handle exceptions. This is done by m_wrapping and forwarding the actual
/// 'PatternRewriter' but overriding its 'create' calls to create exception
/// handling versions of ops if the exception handling version of the operation
/// is being processed.
class ExceptionRewriter final : public mlir::PatternRewriter {
  mlir::Block* m_exceptionHandler = nullptr;
  llvm::SmallVector<mlir::Value> m_exceptionOperands;
  PatternRewriter& m_wrapping;

public:
  ExceptionRewriter(PatternRewriter& wrapper, mlir::Block* exceptionHandler,
                    mlir::ValueRange exceptionOperands)
      : PatternRewriter(static_cast<OpBuilder&>(wrapper)),
        m_exceptionHandler(exceptionHandler),
        m_exceptionOperands(exceptionOperands), m_wrapping(wrapper) {
    setListener(nullptr);
  }

  /// Redefinition to use the 'create' of 'ExceptionRewriter' rather than
  /// 'PatternRewriter's.
  template <typename OpTy, typename... Args>
  auto replaceOpWithNewOp(mlir::Operation* op, Args&&... args) {
    auto newOp = create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
    replaceOp(op, newOp);
    return newOp;
  }

  /// Creates 'OpTy' with 'args' or its exception handling version if processing
  /// an exception handling operation.
  template <typename OpTy, typename... Args>
  auto create(mlir::Location loc, Args&&... args) {
    m_wrapping.setInsertionPoint(getInsertionBlock(), getInsertionPoint());
    if constexpr (std::is_base_of_v<
                      Py::AddableExceptionHandlingInterface::Trait<OpTy>,
                      OpTy>) {
      auto doReturn = [](mlir::Operation* op) {
        if constexpr (OpTy::template hasTrait<mlir::OpTrait::OneResult>())
          return op->getResult(0);
        else
          return op;
      };

      // Nothing to do if there is no exception handler.
      if (!m_exceptionHandler)
        return doReturn(
            m_wrapping.create<OpTy>(loc, std::forward<Args>(args)...));

      // Create the op but do so with this create rather than 'm_wrapping's to
      // avoid the conversion infrastructure of being notified of the transient
      // operation.
      auto op = PatternRewriter::create<OpTy>(loc, std::forward<Args>(args)...);
      // Split the block *after* the operation to make 'op' the terminator.
      mlir::Block* happyPath =
          splitBlock(op->getBlock(), mlir::Block::iterator(op->getNextNode()));
      // After the split the insertion point has moved to the beginning of
      // 'happyPath'. Move it to after the 'op'.
      m_wrapping.setInsertionPointAfter(op);
      // Create the exception handling version from 'op'. This is done with
      // 'm_wrapping' to notify the conversion infrastructure.
      mlir::Operation* newOp = op.cloneWithExceptionHandling(
          m_wrapping, happyPath, m_exceptionHandler, m_exceptionOperands);
      op->erase();

      // Continue in the just created 'happyPath'.
      setInsertionPointToStart(happyPath);
      return doReturn(newOp);
    } else {
      return m_wrapping.create<OpTy>(loc, std::forward<Args>(args)...);
    }
  }

  //===--------------------------------------------------------------------===//
  // Forwarding implementations.
  //===--------------------------------------------------------------------===//

  bool canRecoverFromRewriteFailure() const override {
    return m_wrapping.canRecoverFromRewriteFailure();
  }

  void replaceOp(mlir::Operation* op, mlir::ValueRange newValues) override {
    m_wrapping.replaceOp(op, newValues);
  }

  void replaceOp(mlir::Operation* op, mlir::Operation* newOp) override {
    m_wrapping.replaceOp(op, newOp);
  }

  void eraseOp(mlir::Operation* op) override {
    m_wrapping.eraseOp(op);
  }

  void eraseBlock(mlir::Block* block) override {
    m_wrapping.eraseBlock(block);
  }

  void inlineBlockBefore(mlir::Block* source, mlir::Block* dest,
                         mlir::Block::iterator before,
                         mlir::ValueRange argValues) override {
    m_wrapping.inlineBlockBefore(source, dest, before, argValues);
  }

  void startOpModification(mlir::Operation* op) override {
    m_wrapping.startOpModification(op);
  }

  void finalizeOpModification(mlir::Operation* op) override {
    m_wrapping.finalizeOpModification(op);
  }

  void cancelOpModification(mlir::Operation* op) override {
    m_wrapping.cancelOpModification(op);
  }
};

/// 'RewritePattern' that allows reusing the same 'matchAndRewrite'
/// implementation for both the normal and exception-handling variant of an
/// operation.
/// Derived-classes should implement:
///
/// LogicalResult
/// matchAndRewrite(Interface op, ExceptionRewriter& rewriter) const override {
///   ...
/// }
///
/// Where 'Interface' is the interface derived from the accessors of 'NormalOp'
/// and its exception handling version.
/// 'ExceptionRewriter' is always used even for 'NormalOp' for consistency.
///
/// The only constraint given is that the rewriter must be written in a way that
/// 'op' can be replaced with a branch to the normal destination if handling the
/// exception handling version. The default insertion point is before the
/// operation which satisfies this behaviour. Do not place operations after
/// 'op'!
template <class Interface>
struct OpExRewritePattern : public mlir::OpInterfaceRewritePattern<Interface> {
  using Base = OpExRewritePattern<Interface>;

  OpExRewritePattern(mlir::MLIRContext* context,
                     mlir::PatternBenefit benefit = 1)
      : mlir::OpInterfaceRewritePattern<Interface>(context, benefit) {
    this->setHasBoundedRewriteRecursion();
  }

  using mlir::OpInterfaceRewritePattern<Interface>::matchAndRewrite;

  mlir::LogicalResult
  matchAndRewrite(Interface op, mlir::PatternRewriter& rewriter) const final {
    if (auto invokeOp = llvm::dyn_cast<ExceptionHandlingInterface>(*op)) {
      ExceptionRewriter exceptionRewriter{rewriter, invokeOp.getExceptionPath(),
                                          invokeOp.getUnwindDestOperands()};
      {
        mlir::OpBuilder::InsertionGuard guard{rewriter};
        rewriter.setInsertionPointAfter(op);

        mlir::Location loc = op->getLoc();
        mlir::Block* happyPath = invokeOp.getHappyPath();
        llvm::SmallVector<mlir::Value> happyOperands(
            invokeOp.getNormalDestOperands());
        rewriter.create<mlir::cf::BranchOp>(loc, happyPath, happyOperands);
      }
      return this->matchAndRewrite(op, exceptionRewriter);
    }

    ExceptionRewriter exceptionRewriter{rewriter, nullptr, {}};
    return this->matchAndRewrite(op, exceptionRewriter);
  }

  virtual mlir::LogicalResult
  matchAndRewrite(Interface op, ExceptionRewriter& rewriter) const = 0;
};

} // namespace pylir::Py
