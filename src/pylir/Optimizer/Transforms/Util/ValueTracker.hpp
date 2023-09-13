//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <utility>

namespace pylir {
class ValueTracker {
  mlir::OwningOpRef<mlir::UnrealizedConversionCastOp> m_tracker;

  [[nodiscard]] mlir::OwningOpRef<mlir::UnrealizedConversionCastOp>&
  tracker() const {
    return const_cast<std::remove_const_t<decltype(m_tracker)>&>(m_tracker);
  }

public:
  ValueTracker() = default;

  ~ValueTracker() = default;

  ValueTracker(const ValueTracker& rhs)
      : m_tracker(rhs.m_tracker ? rhs.tracker()->clone() : nullptr) {}

  ValueTracker& operator=(const ValueTracker& rhs) {
    if (this == &rhs)
      return *this;

    if (m_tracker)
      m_tracker->erase();

    m_tracker = rhs.m_tracker ? rhs.tracker()->clone() : nullptr;
    return *this;
  }

  ValueTracker(ValueTracker&& rhs) noexcept = default;

  ValueTracker& operator=(ValueTracker&& rhs) noexcept = default;

  /*implicit*/ ValueTracker(mlir::Value value) {
    if (!value)
      return;

    mlir::OpBuilder builder(value.getContext());
    m_tracker = builder.create<mlir::UnrealizedConversionCastOp>(
        builder.getUnknownLoc(), mlir::TypeRange{}, value);
  }

  ValueTracker& operator=(mlir::Value value) {
    if (!m_tracker || !value)
      return *this = ValueTracker(value);

    m_tracker->getInputsMutable().assign(value);
    return *this;
  }

  /*implicit*/ operator mlir::Value() const {
    if (!m_tracker)
      return {};

    return tracker()->getInputs()[0];
  }

  friend bool operator==(mlir::Value value, const ValueTracker& valueTracker) {
    return static_cast<mlir::Value>(valueTracker) == value;
  }

  friend bool operator==(const ValueTracker& valueTracker, mlir::Value value) {
    return static_cast<mlir::Value>(valueTracker) == value;
  }
};
} // namespace pylir
