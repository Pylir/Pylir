//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>
#include <pylir/Optimizer/Interfaces/MemoryFoldInterface.hpp>
#include <pylir/Optimizer/Interfaces/SROAInterfaces.hpp>
#include <pylir/Optimizer/PylirPy/IR/TypeRefineableInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/CopyObjectInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/ExceptionHandlingInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/OnlyReadsValueInterface.hpp>

#include <optional>
#include <variant>

#include "PylirPyAttributes.hpp"
#include "PylirPyTraits.hpp"
#include "PylirPyTypes.hpp"

namespace pylir::Py
{
struct IterExpansion
{
    mlir::Value value;
};

struct MappingExpansion
{
    mlir::Value value;
};

struct DictEntry
{
    mlir::Value key;
    mlir::Value hash;
    mlir::Value value;
};

using DictArg = std::variant<DictEntry, MappingExpansion>;
using IterArg = std::variant<mlir::Value, IterExpansion>;

enum class OperandShape
{
    Single,
    Variadic,
};

namespace details
{
mlir::Operation* cloneWithExceptionHandlingImpl(mlir::OpBuilder& builder, mlir::Operation* operation,
                                                const mlir::OperationName& invokeVersion, ::mlir::Block* happyPath,
                                                mlir::Block* exceptionPath, mlir::ValueRange unwindOperands,
                                                llvm::StringRef attrSizedSegmentName,
                                                llvm::ArrayRef<OperandShape> shape);
} // namespace details

template <class InvokeVersion, OperandShape... shape>
struct AddableExceptionHandling
{
    template <class ConcreteType>
    class Impl : public AddableExceptionHandlingInterface::Trait<ConcreteType>
    {
        template <unsigned n>
        constexpr static std::optional<unsigned> checkNOperands()
        {
            if constexpr (n == 1)
            {
                return std::nullopt;
            }
            else if constexpr (ConcreteType::template hasTrait<mlir::OpTrait::NOperands<n>::template Impl>())
            {
                return n;
            }
            else
            {
                return checkNOperands<n - 1>();
            }
        }

        constexpr static std::optional<unsigned> tryDeduceShape()
        {
            if constexpr (ConcreteType::template hasTrait<mlir::OpTrait::VariadicOperands>())
            {
                return {};
            }
            else if constexpr (ConcreteType::template hasTrait<mlir::OpTrait::OneOperand>())
            {
                return 1;
            }
            else
            {
                return checkNOperands<5>();
            }
        }

    public:
        mlir::Operation* cloneWithExceptionHandling(mlir::OpBuilder& builder, ::mlir::Block* happyPath,
                                                    mlir::Block* exceptionPath, mlir::ValueRange unwindOperands)
        {
            constexpr auto deduced = tryDeduceShape();
            static_assert(ConcreteType::template hasTrait<mlir::OpTrait::AttrSizedOperandSegments>()
                              || ConcreteType::template hasTrait<mlir::OpTrait::VariadicOperands>()
                              || deduced.has_value() || sizeof...(shape) > 0,
                          "Could not deduce shape of the operations operands, nor was it explicitly specified");
            constexpr auto shapeSize =
                deduced.has_value() ?
                    *deduced :
                    (ConcreteType::template hasTrait<mlir::OpTrait::VariadicOperands>() ? 1 : sizeof...(shape));
            std::array<OperandShape, shapeSize> result;
            if constexpr (ConcreteType::template hasTrait<mlir::OpTrait::VariadicOperands>())
            {
                result[0] = OperandShape::Variadic;
            }
            else if constexpr (deduced.has_value())
            {
                std::fill(result.begin(), result.end(), OperandShape::Single);
            }
            else
            {
                auto initList = {shape...};
                std::copy(initList.begin(), initList.end(), result.begin());
            }
            return details::cloneWithExceptionHandlingImpl(
                builder, this->getOperation(),
                mlir::OperationName(InvokeVersion::getOperationName(), builder.getContext()), happyPath, exceptionPath,
                unwindOperands, mlir::OpTrait::AttrSizedOperandSegments<InvokeVersion>::getOperandSegmentSizeAttr(),
                result);
        }
    };
};

class DictArgsIterator
{
    mlir::OperandRange::iterator m_keys;
    mlir::OperandRange::iterator m_hashes;
    mlir::OperandRange::iterator m_values;
    llvm::ArrayRef<std::int32_t> m_expansions;
    llvm::ArrayRef<std::int32_t>::iterator m_currExp;
    std::int32_t m_index = 0;

    bool isCurrentlyExpansion();

public:
    using value_type = DictArg;
    using reference = DictArg;
    using pointer = DictArg*;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;

    DictArgsIterator(mlir::OperandRange::iterator keys, mlir::OperandRange::iterator hashes,
                     mlir::OperandRange::iterator values, llvm::ArrayRef<std::int32_t>::iterator expIterator,
                     llvm::ArrayRef<std::int32_t> expansions, std::size_t index)
        : m_keys(keys),
          m_hashes(hashes),
          m_values(values),
          m_expansions(expansions),
          m_currExp(expIterator),
          m_index(static_cast<std::int32_t>(index))
    {
    }

    DictArg operator*();

    bool operator==(const DictArgsIterator& rhs) const
    {
        return m_keys == rhs.m_keys;
    }

    bool operator!=(const DictArgsIterator& rhs) const
    {
        return !(rhs == *this);
    }

    DictArgsIterator& operator++();

    DictArgsIterator operator++(int)
    {
        auto copy = *this;
        ++(*this);
        return copy;
    }

    DictArgsIterator& operator--();

    DictArgsIterator operator--(int)
    {
        auto copy = *this;
        --(*this);
        return copy;
    }
};

#define TRIVIAL_RESOURCE(prefix)                                                  \
    struct prefix##Resource : mlir::SideEffects::Resource::Base<prefix##Resource> \
    {                                                                             \
        llvm::StringRef getName() final                                           \
        {                                                                         \
            return #prefix "Resource";                                            \
        }                                                                         \
    }

/// Reads and writes from 'py.globalHandle'.
TRIVIAL_RESOURCE(Handle);

/// Reads and writes to the object parts. This currently just reads and writes to slots.
TRIVIAL_RESOURCE(Object);

/// Reads and writes to the list parts of a list object.
TRIVIAL_RESOURCE(List);

/// Reads and writes to the dict parts of a dict object.
TRIVIAL_RESOURCE(Dict);

#undef TRIVIAL_RESOURCE

inline auto getAllResources()
{
    return std::array<mlir::SideEffects::Resource*, 4>{HandleResource::get(), ObjectResource::get(),
                                                       ListResource::get(), DictResource::get()};
}

} // namespace pylir::Py

#include <pylir/Optimizer/PylirPy/IR/PylirPyEnums.h.inc>

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.h.inc>
