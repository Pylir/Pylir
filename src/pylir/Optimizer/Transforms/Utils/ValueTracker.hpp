#pragma once

#include <mlir/IR/Operation.h>

#include <utility>

namespace pylir
{
class ValueTracker
{
    mlir::Operation* m_tracker = nullptr;

public:
    ValueTracker() = default;

    ~ValueTracker()
    {
        if (m_tracker)
        {
            m_tracker->erase();
        }
    }

    ValueTracker(const ValueTracker& rhs) : m_tracker(rhs.m_tracker ? rhs.m_tracker->clone() : nullptr) {}

    ValueTracker& operator=(const ValueTracker& rhs)
    {
        if (m_tracker)
        {
            m_tracker->erase();
        }
        m_tracker = rhs.m_tracker ? rhs.m_tracker->clone() : nullptr;
        return *this;
    }

    ValueTracker(ValueTracker&& rhs) noexcept : m_tracker(std::exchange(rhs.m_tracker, nullptr)) {}

    ValueTracker& operator=(ValueTracker&& rhs) noexcept
    {
        if (m_tracker)
        {
            m_tracker->erase();
        }
        m_tracker = std::exchange(rhs.m_tracker, nullptr);
        return *this;
    }

    ValueTracker(mlir::Value value)
    {
        if (!value)
        {
            return;
        }
        auto* context = value.getContext();
        context->allowUnregisteredDialects();
        mlir::OperationState state(mlir::UnknownLoc::get(context), "__value_tracker");
        state.addOperands(value);
        m_tracker = mlir::Operation::create(state);
    }

    ValueTracker& operator=(mlir::Value value)
    {
        if (!m_tracker || !value)
        {
            return *this = ValueTracker(value);
        }
        m_tracker->setOperand(0, value);
        return *this;
    }

    operator mlir::Value() const
    {
        if (!m_tracker)
        {
            return {};
        }
        return m_tracker->getOperand(0);
    }
};
} // namespace pylir
