//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>

#include <pylir/Interfaces/Builtins.hpp>
#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>
#include <pylir/Optimizer/Interfaces/MemoryFoldInterface.hpp>
#include <pylir/Optimizer/PylirPy/IR/TypeRefineableInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/ObjectFromTypeObjectInterface.hpp>

#include "PylirPyTypes.hpp"

namespace pylir::Py
{

template <class ConcreteType>
class AlwaysBound : public mlir::OpTrait::TraitBase<ConcreteType, AlwaysBound>
{
    static mlir::LogicalResult verifyTrait(mlir::Operation*)
    {
        static_assert(!ConcreteType::template hasTrait<mlir::OpTrait::ZeroOperands>(),
                      "'Always Bound' trait is ony applicable to ops with results");
        return mlir::success();
    }
};

template <class ConcreteType>
class ReturnsImmutable : public mlir::OpTrait::TraitBase<ConcreteType, ReturnsImmutable>
{
    static mlir::LogicalResult verifyTrait(mlir::Operation*)
    {
        static_assert(!ConcreteType::template hasTrait<mlir::OpTrait::ZeroOperands>(),
                      "'ReturnsImmutable' trait is ony applicable to ops with results");
        return mlir::success();
    }
};

template <const Builtins::Builtin& builtin>
class RefinedType
{
public:
    template <class ConcreteType>
    class Impl : public TypeRefineableInterface::Trait<ConcreteType>
    {
    public:
        pylir::Py::TypeRefineResult refineTypes(llvm::ArrayRef<pylir::Py::TypeAttrUnion>,
                                                llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result)
        {
            auto* context = this->getOperation()->getContext();
            result.emplace_back(pylir::Py::ClassType::get(RefAttr::get(context, builtin.name)));
            return TypeRefineResult::Success;
        }
    };
};

template <class ConcreteType>
class RefinedTypeTupleApproximate : public TypeRefineableInterface::Trait<ConcreteType>
{
public:
    pylir::Py::TypeRefineResult refineTypes(llvm::ArrayRef<pylir::Py::TypeAttrUnion>,
                                            llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result)
    {
        result.emplace_back(Py::ClassType::get(RefAttr::get(this->getOperation()->getContext(), Builtins::Tuple.name)));
        return TypeRefineResult::Approximate;
    }
};

template <class ConcreteType>
class RefinedObjectFromTypeObjectImpl : public TypeRefineableInterface::Trait<ConcreteType>
{
public:
    pylir::Py::TypeRefineResult refineTypes(llvm::ArrayRef<pylir::Py::TypeAttrUnion> operands,
                                            llvm::SmallVectorImpl<pylir::Py::ObjectTypeInterface>& result)
    {
        RefAttr type = operands[mlir::cast<ConcreteType>(this->getOperation()).getTypeObjectIndex()]
                           .template dyn_cast_or_null<RefAttr>();
        if (!type)
        {
            return TypeRefineResult::Failure;
        }
        result.emplace_back(Py::ClassType::get(type));
        return TypeRefineResult::Success;
    }
};

template <class ConcreteType>
class ImmutableAttr : public mlir::AttributeTrait::TraitBase<ConcreteType, ImmutableAttr>
{
};

} // namespace pylir::Py
