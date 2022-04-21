// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <utility>

namespace pylir
{
class ValueTracker
{
    mlir::OwningOpRef<mlir::UnrealizedConversionCastOp> m_tracker;

    mlir::OwningOpRef<mlir::UnrealizedConversionCastOp>& tracker() const
    {
        return const_cast<std::remove_const_t<decltype(m_tracker)>&>(m_tracker);
    }

public:
    ValueTracker() = default;

    ValueTracker(const ValueTracker& rhs) : m_tracker(rhs.m_tracker ? rhs.tracker()->clone() : nullptr) {}

    ValueTracker& operator=(const ValueTracker& rhs)
    {
        if (this == &rhs)
        {
            return *this;
        }
        if (m_tracker)
        {
            m_tracker->erase();
        }
        m_tracker = rhs.m_tracker ? rhs.tracker()->clone() : nullptr;
        return *this;
    }

    ValueTracker(ValueTracker&& rhs) noexcept = default;

    ValueTracker& operator=(ValueTracker&& rhs) noexcept = default;

    // NOLINTNEXTLINE(google-explicit-constructor)
    ValueTracker(mlir::Value value)
    {
        if (!value)
        {
            return;
        }
        mlir::OpBuilder builder(value.getContext());
        m_tracker = builder.create<mlir::UnrealizedConversionCastOp>(builder.getUnknownLoc(), mlir::TypeRange{}, value);
    }

    ValueTracker& operator=(mlir::Value value)
    {
        if (!m_tracker || !value)
        {
            return *this = ValueTracker(value);
        }
        m_tracker->getInputsMutable().assign(value);
        return *this;
    }

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator mlir::Value() const
    {
        if (!m_tracker)
        {
            return {};
        }
        return tracker()->getInputs()[0];
    }
};
} // namespace pylir
