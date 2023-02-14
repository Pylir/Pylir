//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <pylir/Interfaces/Builtins.hpp>
#include <pylir/Optimizer/Interfaces/SROAInterfaces.hpp>
#include <pylir/Support/BigInt.hpp>

#include <map>

#include "ObjectAttrInterface.hpp"
#include "PylirPyTraits.hpp"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyAttributes.h.inc"

namespace pylir::Py
{

class BoolAttr : public IntAttr
{
public:
    using IntAttr::IntAttr;

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return {"bool"};
    }

    static BoolAttr get(mlir::MLIRContext* context, bool value)
    {
        return IntAttr::get(context, BigInt(value), RefAttr::get(context, Builtins::Bool.name)).cast<BoolAttr>();
    }

    static mlir::Attribute parse(mlir::AsmParser& parser, mlir::Type type);

    void print(mlir::AsmPrinter& printer) const;

    static bool classof(mlir::Attribute attr)
    {
        auto intAttr = attr.dyn_cast<IntAttr>();
        // Since python does not allow 'bool' to be subclassed, this is sufficient.
        // If we were to allow it, this would have to change.
        return intAttr && intAttr.getTypeObject().getRef().getValue() == Builtins::Bool.name;
    }

    [[nodiscard]] bool getValue() const
    {
        return !IntAttr::getValue().isZero();
    }
};

} // namespace pylir::Py

template <class First, class Second>
struct mlir::AttrTypeSubElementHandler<std::pair<First, Second>,
                                       std::enable_if_t<mlir::has_sub_attr_or_type_v<First, Second>>>
{
    static void walk(const std::pair<First, Second>& param, AttrTypeImmediateSubElementWalker& walker)
    {
        AttrTypeSubElementHandler<First>::walk(param.first, walker);
        AttrTypeSubElementHandler<Second>::walk(param.second, walker);
    }

    static std::pair<First, Second> replace(const std::pair<First, Second>& param,
                                            AttrSubElementReplacements& attrRepls,
                                            TypeSubElementReplacements& typeRepls)
    {
        return {AttrTypeSubElementHandler<First>::replace(param.first, attrRepls, typeRepls),
                AttrTypeSubElementHandler<Second>::replace(param.second, attrRepls, typeRepls)};
    }
};