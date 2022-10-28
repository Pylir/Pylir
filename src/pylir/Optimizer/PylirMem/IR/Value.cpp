//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Value.hpp"

#include <mlir/IR/Matchers.h>

#include "pylir/Optimizer/PylirPy/IR/Value.hpp"

llvm::Optional<pylir::Mem::LayoutType> pylir::Mem::getLayoutType(mlir::Value value,
                                                                 llvm::DenseMap<mlir::Attribute, LayoutType>* cache)
{
    mlir::Attribute attribute;
    if (mlir::matchPattern(value, mlir::m_Constant(&attribute)))
    {
        return getLayoutType(attribute, cache);
    }
    // TODO: Future deduction from some kind of "makeType" op goes here.
    return llvm::None;
}

namespace
{
llvm::Optional<pylir::Mem::LayoutType> getLayoutTypeImpl(mlir::Attribute attr)
{
    using namespace pylir;
    using namespace pylir::Py;
    using namespace pylir::Mem;

    auto mapLayoutType = [](mlir::Attribute attribute) -> llvm::Optional<pylir::Mem::LayoutType>
    {
        auto ref = attribute.dyn_cast<RefAttr>();
        if (!ref)
        {
            return llvm::None;
        }
        return llvm::StringSwitch<llvm::Optional<LayoutType>>(ref.getRef().getValue())
            .Case(Builtins::Object.name, LayoutType::Object)
            .Case(Builtins::Type.name, LayoutType::Type)
            .Case(Builtins::Float.name, LayoutType::Float)
            .Case(Builtins::Function.name, LayoutType::Function)
            .Case(Builtins::Tuple.name, LayoutType::Tuple)
            .Case(Builtins::List.name, LayoutType::List)
            .Case(Builtins::Str.name, LayoutType::String)
            .Case(Builtins::Dict.name, LayoutType::Dict)
            .Case(Builtins::Int.name, LayoutType::Int)
            .Case(Builtins::BaseException.name, LayoutType::BaseException)
            .Default(llvm::None);
    };

    if (auto result = mapLayoutType(attr))
    {
        return result;
    }

    auto type = ref_cast<TypeAttr>(attr);
    if (!type)
    {
        return llvm::None;
    }
    auto mro = ref_cast<TupleAttr>(type.getMroTuple());
    for (mlir::Attribute iter : mro)
    {
        if (auto result = mapLayoutType(iter))
        {
            return result;
        }
    }
    return LayoutType::Object;
}
} // namespace

llvm::Optional<pylir::Mem::LayoutType> pylir::Mem::getLayoutType(mlir::Attribute attr,
                                                                 llvm::DenseMap<mlir::Attribute, LayoutType>* cache)
{
    if (cache)
    {
        if (auto res = cache->find(attr); res != cache->end())
        {
            return res->second;
        }
    }

    auto result = getLayoutTypeImpl(attr);
    if (result && cache)
    {
        (*cache)[attr] = *result;
    }
    return result;
}

pylir::Py::RefAttr pylir::Mem::layoutTypeToTypeObject(mlir::MLIRContext* context, pylir::Mem::LayoutType layoutType)
{
    switch (layoutType)
    {
        case pylir::Mem::LayoutType::Object: return pylir::Py::RefAttr::get(context, pylir::Builtins::Object.name);
        case pylir::Mem::LayoutType::Type: return pylir::Py::RefAttr::get(context, pylir::Builtins::Type.name);
        case pylir::Mem::LayoutType::Float: return pylir::Py::RefAttr::get(context, pylir::Builtins::Float.name);
        case pylir::Mem::LayoutType::Function: return pylir::Py::RefAttr::get(context, pylir::Builtins::Function.name);
        case pylir::Mem::LayoutType::Tuple: return pylir::Py::RefAttr::get(context, pylir::Builtins::Tuple.name);
        case pylir::Mem::LayoutType::List: return pylir::Py::RefAttr::get(context, pylir::Builtins::List.name);
        case pylir::Mem::LayoutType::String: return pylir::Py::RefAttr::get(context, pylir::Builtins::Str.name);
        case pylir::Mem::LayoutType::Dict: return pylir::Py::RefAttr::get(context, pylir::Builtins::Dict.name);
        case pylir::Mem::LayoutType::Int: return pylir::Py::RefAttr::get(context, pylir::Builtins::Int.name);
        case pylir::Mem::LayoutType::BaseException:
            return pylir::Py::RefAttr::get(context, pylir::Builtins::BaseException.name);
    }
}