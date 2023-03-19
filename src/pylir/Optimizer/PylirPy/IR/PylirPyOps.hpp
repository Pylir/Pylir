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

#include <llvm/ADT/iterator.h>

#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>
#include <pylir/Optimizer/Interfaces/MemoryFoldInterface.hpp>
#include <pylir/Optimizer/Interfaces/SROAInterfaces.hpp>
#include <pylir/Optimizer/PylirPy/IR/TypeRefineableInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/CopyObjectInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/ExceptionHandlingInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/OnlyReadsValueInterface.hpp>
#include <pylir/Support/Variant.hpp>

#include <optional>
#include <variant>

#include "PylirPyAttributes.hpp"
#include "PylirPyDialect.hpp"
#include "PylirPyTraits.hpp"
#include "PylirPyTypes.hpp"

namespace pylir::Py
{

/// Struct used for the case of a mapping expansion (unary '**' in specific contexts in python). The contained value is
/// mapping being expanded.
struct MappingExpansion
{
    mlir::Value value;

    bool operator==(const MappingExpansion& rhs) const
    {
        return value == rhs.value;
    }

    bool operator!=(const MappingExpansion& rhs) const
    {
        return !(rhs == *this);
    }
};

/// Struct used for normal dictionary entries. Comprised of the three operands which are the key, the keys hash and the
/// value respectively.
struct DictEntry
{
    mlir::Value key;
    mlir::Value hash;
    mlir::Value value;

    bool operator==(const DictEntry& rhs) const
    {
        return std::tie(key, hash, value) == std::tie(rhs.key, rhs.hash, rhs.value);
    }

    bool operator!=(const DictEntry& rhs) const
    {
        return !(rhs == *this);
    }
};

/// Variant used when interacting with 'py.makeDict' and 'py.makeDictEx' to represent operands that may either be an
/// expansion or a normal entry.
using DictArg = std::variant<DictEntry, MappingExpansion>;

class MakeDictOp;
class MakeDictExOp;

/// Bidirectional iterator over the dictionary operands of 'py.makeDict' and 'py.makeDictEx'. Returns a 'DictArg'
/// which may either be a 'MappingExpansion' or 'DictEntry'.
class DictArgsIterator : public llvm::iterator_facade_base<DictArgsIterator, std::bidirectional_iterator_tag, DictArg,
                                                           std::ptrdiff_t, DictArg*, DictArg>
{
    llvm::PointerUnion<MakeDictOp, MakeDictExOp> m_op;
    llvm::ArrayRef<std::int32_t>::iterator m_currExp{};
    std::int32_t m_keyIndex = 0;
    std::int32_t m_valueIndex = 0;

    bool isCurrentlyExpansion();

    DictArgsIterator(llvm::PointerUnion<MakeDictOp, MakeDictExOp> op) : m_op(op)
    {
        PYLIR_ASSERT(m_op);
    }

    llvm::ArrayRef<std::int32_t> getExpansion() const;

    [[nodiscard]] mlir::OperandRange getKeys() const;

    [[nodiscard]] mlir::OperandRange getValues() const;

    [[nodiscard]] mlir::OperandRange getHashes() const;

public:
    static DictArgsIterator begin(llvm::PointerUnion<MakeDictOp, MakeDictExOp> op);

    static DictArgsIterator end(llvm::PointerUnion<MakeDictOp, MakeDictExOp> op);

    DictArg operator*();

    bool operator==(const DictArgsIterator& rhs) const
    {
        return m_keyIndex == rhs.m_keyIndex;
    }

    DictArgsIterator& operator++();

    DictArgsIterator& operator--();
};

struct IterExpansion
{
    mlir::Value value;
};

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
            std::array<OperandShape, shapeSize> result{};
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

#define TRIVIAL_RESOURCE(prefix)                                                  \
    struct prefix##Resource : mlir::SideEffects::Resource::Base<prefix##Resource> \
    {                                                                             \
        llvm::StringRef getName() final                                           \
        {                                                                         \
            return #prefix "Resource";                                            \
        }                                                                         \
    }

/// Reads and writes from 'py.global'.
TRIVIAL_RESOURCE(Global);

/// Reads and writes to the object parts. This currently just reads and writes to slots.
TRIVIAL_RESOURCE(Object);

/// Reads and writes to the list parts of a list object.
TRIVIAL_RESOURCE(List);

/// Reads and writes to the dict parts of a dict object.
TRIVIAL_RESOURCE(Dict);

#undef TRIVIAL_RESOURCE

inline auto getAllResources()
{
    return std::array<mlir::SideEffects::Resource*, 4>{GlobalResource::get(), ObjectResource::get(),
                                                       ListResource::get(), DictResource::get()};
}

} // namespace pylir::Py

#include <pylir/Optimizer/PylirPy/IR/PylirPyEnums.h.inc>

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.h.inc>

#define SUPPORT_OP_IN_POINTER_UNION(opType)                                                                      \
    template <>                                                                                                  \
    struct llvm::PointerLikeTypeTraits<opType>                                                                   \
    {                                                                                                            \
        static void* getAsVoidPointer(opType op)                                                                 \
        {                                                                                                        \
            return const_cast<void*>(op.getAsOpaquePointer());                                                   \
        }                                                                                                        \
                                                                                                                 \
        static opType getFromVoidPointer(void* p)                                                                \
        {                                                                                                        \
            return opType::getFromOpaquePointer(p);                                                              \
        }                                                                                                        \
                                                                                                                 \
        constexpr static int NumLowBitsAvailable = PointerLikeTypeTraits<mlir::Operation*>::NumLowBitsAvailable; \
    }

SUPPORT_OP_IN_POINTER_UNION(pylir::Py::MakeDictOp);
SUPPORT_OP_IN_POINTER_UNION(pylir::Py::MakeDictExOp);

#undef SUPPORT_OP_IN_POINTER_UNION
