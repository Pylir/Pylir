// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>

#include <pylir/Optimizer/Interfaces/CaptureInterface.hpp>
#include <pylir/Optimizer/Interfaces/MemoryFoldInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/ObjectFromTypeObjectInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/TypeRefineableInterface.hpp>

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

template <class ConcreteType>
class NoCapture : public CaptureInterface::Trait<ConcreteType>
{
public:
    bool capturesOperand(unsigned int)
    {
        return false;
    }
};

#define BUILTIN_TYPE(x, ...)                                                                                          \
    template <class ConcreteType>                                                                                     \
    class x##RefinedType : public TypeRefineableInterface::Trait<ConcreteType>                                        \
    {                                                                                                                 \
    public:                                                                                                           \
        llvm::SmallVector<pylir::Py::ObjectTypeInterface> refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface>, \
                                                                      mlir::SymbolTable*)                             \
        {                                                                                                             \
            auto* context = this->getOperation()->getContext();                                                       \
            return {pylir::Py::ClassType::get(                                                                        \
                context, mlir::FlatSymbolRefAttr::get(context, pylir::Py::Builtins::x.name), llvm::None)};            \
        }                                                                                                             \
    };

#include <pylir/Interfaces/Builtins.def>

template <class ConcreteType>
class RefinedObjectFromTypeObject : public TypeRefineableInterface::Trait<ConcreteType>
{
public:
    llvm::SmallVector<pylir::Py::ObjectTypeInterface> refineTypes(llvm::ArrayRef<pylir::Py::ObjectTypeInterface>,
                                                                  mlir::SymbolTable*)
    {
        mlir::FlatSymbolRefAttr type;
        if (!mlir::matchPattern(mlir::cast<ConcreteType>(this->getOperation()).getTypeObject(),
                                mlir::m_Constant(&type)))
        {
            return {Py::UnknownType::get(this->getOperation()->getContext())};
        }
        return {Py::ClassType::get(this->getOperation()->getContext(), type, llvm::None)};
    }
};

} // namespace pylir::Py
