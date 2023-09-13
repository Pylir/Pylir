//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Value.h>

#include <llvm/Support/Allocator.h>

namespace pylir {
/// Base class for all patterns that are created by
/// 'DialectImplicationPatternsInterface::getImplicationPatterns'. These
/// patterns are created by matching a conditional and its given value on a path
/// in the control flow graph. The pattern is then active within that path.
/// Patterns may also have state.
///
/// Note: Patterns are currently required to be trivially destructible by the
/// 'PatternAllocator'. This also necessitates patterns not having a virtual
/// destructor. You can disable warnings by the compiler about not having a
/// trivial destructors by marking the class 'final'.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class ImplicationPatternBase {
public:
  /// Function called by the pass to allow the pattern to attempt to rewrite the
  /// given operation. For creating new operations, 'builder' may be used. The
  /// function MUST return 'success' if it it did any modifications to the IR
  /// and MUST return 'failure' if it did not.
  virtual mlir::LogicalResult
  matchAndRewrite(mlir::Operation* operation,
                  mlir::OpBuilder& builder) const = 0;
};

#pragma clang diagnostic pop

/// Allocator use for allocating Patterns and giving them a stable lifetime
/// throughout the lifetime of the pass.
class PatternAllocator {
  llvm::BumpPtrAllocator m_allocator;

public:
  /// Allocates the storage for a given pattern 'T', constructs it with the
  /// given 'args' and returns a pointer to the newly allocated pattern.
  template <class T, class... Args>
  T* allocate(Args&&... args) {
    static_assert(std::is_trivially_destructible_v<T>);
    return new (m_allocator.Allocate<T>()) T{std::forward<Args>(args)...};
  }

  /// Convenience function for creating a pattern from a callable class (most
  /// commonly a lambda). The callable must be callable with the same signature
  /// as 'ImplicationPatternBase::matchAndRewrite'.
  template <class F, std::enable_if_t<std::is_class_v<F>>* = nullptr>
  auto* allocate(F&& f) {
    using Lambda = std::decay_t<F>;
    struct LambdaAdaptor final : Lambda, ImplicationPatternBase {
      explicit LambdaAdaptor(F&& f) : Lambda(std::forward<F>(f)) {}

      mlir::LogicalResult
      matchAndRewrite(mlir::Operation* operation,
                      mlir::OpBuilder& builder) const override {
        return (*this)(operation, builder);
      }
    };
    return allocate<LambdaAdaptor>(std::forward<F>(f));
  }
};

/// Dialect interface for a dialect to supply patterns and further implications
/// to a pass. Passes will load all implementations of this interface for all
/// loaded dialects in the context.
class DialectImplicationPatternsInterface
    : public mlir::DialectInterface::Base<DialectImplicationPatternsInterface> {
public:
  explicit DialectImplicationPatternsInterface(mlir::Dialect* dialect)
      : Base(dialect) {}

  /// Hook used by the pass to allow dialects to create patterns or more
  /// implications for a given implication within a path in the CFG.
  ///
  /// 'conditional' is the IR value object whose value in a given path has been
  /// implied to be the constant 'value'. 'patternAddCallback' should be used to
  /// add patterns active within that path to allow rewriting the IR. The
  /// patterns must have been allocated with 'allocator'.
  /// 'implicationAddCallback' may be used to add more implications for a given
  /// path in the CFG.
  ///
  /// One use for 'implicationAddCallback' is for eg '%conditional = not
  /// %otherValue', where '%conditional' has been given the constant value
  /// 'true' for a path, to allow also implying that '%otherValue' must be false
  /// in that path.
  virtual void getImplicationPatterns(
      PatternAllocator& allocator, mlir::Value conditional,
      mlir::Attribute value,
      llvm::function_ref<void(ImplicationPatternBase*)> patternAddCallback,
      llvm::function_ref<void(mlir::Value, mlir::Attribute)>
          implicationAddCallback) const = 0;
};

} // namespace pylir
