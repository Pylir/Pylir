// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirPyAttributes.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <utility>

#include "PylirPyDialect.hpp"

template <>
struct mlir::FieldParser<llvm::APFloat>
{
    static mlir::FailureOr<llvm::APFloat> parse(mlir::AsmParser& parser)
    {
        double value;
        if (parser.parseFloat(value))
        {
            return mlir::failure();
        }
        return llvm::APFloat(value);
    }
};

template <>
struct mlir::FieldParser<pylir::BigInt>
{
    static mlir::FailureOr<pylir::BigInt> parse(mlir::AsmParser& parser)
    {
        llvm::APInt apInt;
        if (parser.parseInteger(apInt))
        {
            return mlir::failure();
        }
        llvm::SmallString<10> str;
        apInt.toStringSigned(str);
        return pylir::BigInt(std::string{str.data(), str.size()});
    }
};

namespace pylir
{
llvm::hash_code hash_value(const pylir::BigInt& bigInt)
{
    auto count = mp_sbin_size(&bigInt.getHandle());
    llvm::SmallVector<std::uint8_t, 10> data(count);
    auto result = mp_to_sbin(&bigInt.getHandle(), data.data(), count, nullptr);
    PYLIR_ASSERT(result == MP_OKAY);
    return llvm::hash_value(makeArrayRef(data));
}
} // namespace pylir

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.cpp.inc"

void pylir::Py::PylirPyDialect::initializeAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.cpp.inc"
        >();
}

const pylir::BigInt& pylir::Py::IntAttr::getIntegerValue() const
{
    return getValue();
}

const pylir::BigInt& pylir::Py::BoolAttr::getIntegerValue() const
{
    static pylir::BigInt trueValue(1);
    static pylir::BigInt falseValue(0);
    return getValue() ? trueValue : falseValue;
}

mlir::FlatSymbolRefAttr pylir::Py::FunctionAttr::getTypeObject() const
{
    return mlir::FlatSymbolRefAttr::get(getContext(), Builtins::Function.name);
}

mlir::DictionaryAttr pylir::Py::FunctionAttr::getSlots() const
{
    llvm::SmallVector<mlir::NamedAttribute> vector = {
        mlir::NamedAttribute(mlir::StringAttr::get(getContext(), "__defaults__"), getDefaults()),
    };
    if (getDict())
    {
        vector.emplace_back(mlir::NamedAttribute(mlir::StringAttr::get(getContext(), "__dict__"), getDict()));
    }
    vector.emplace_back(mlir::NamedAttribute(mlir::StringAttr::get(getContext(), "__kwdefaults__"), getKwDefaults()));
    vector.emplace_back(mlir::NamedAttribute(mlir::StringAttr::get(getContext(), "__qualname__"), getQualName()));
    return mlir::DictionaryAttr::get(getContext(), vector);
}

void pylir::Py::DictAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                   llvm::function_ref<void(mlir::Type)>) const
{
    std::for_each(getValue().begin(), getValue().end(),
                  [&](auto&& pair)
                  {
                      walkAttrsFn(pair.first);
                      walkAttrsFn(pair.second);
                  });
    walkAttrsFn(getTypeObject());
    walkAttrsFn(getSlots());
}

mlir::Attribute pylir::Py::DictAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                 llvm::ArrayRef<mlir::Type>) const
{
    auto type = replAttrs.take_back(2).back().cast<mlir::FlatSymbolRefAttr>();
    auto slots = replAttrs.back().cast<mlir::DictionaryAttr>();
    std::vector<std::pair<mlir::Attribute, mlir::Attribute>> vector;
    for (std::size_t i = 0; i < replAttrs.size() - 2; i += 2)
    {
        vector.emplace_back(replAttrs[i], replAttrs[i + 1]);
    }
    return get(getContext(), vector, type, slots);
}

namespace
{
template <class Op>
void doTypeObjectSlotsWalk(Op op, llvm::function_ref<void(mlir::Attribute)> walkAttrsFn)
{
    walkAttrsFn(op.getTypeObject());
    walkAttrsFn(op.getSlots());
}

template <class Op, class... Args>
Op doTypeObjectSlotsReplace(Op op, llvm::ArrayRef<mlir::Attribute> replAttrs, Args&&... prior)
{
    auto type = replAttrs.take_back(2).back().cast<mlir::FlatSymbolRefAttr>();
    auto slots = replAttrs.back().cast<mlir::DictionaryAttr>();
    return Op::get(op.getContext(), std::forward<Args>(prior)..., type, slots);
}
} // namespace

void pylir::Py::ObjectAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                     llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::Attribute pylir::Py::ObjectAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                   llvm::ArrayRef<mlir::Type>) const
{
    return doTypeObjectSlotsReplace(*this, replAttrs);
}

void pylir::Py::IntAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                  llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::Attribute pylir::Py::IntAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                llvm::ArrayRef<mlir::Type>) const
{
    return doTypeObjectSlotsReplace(*this, replAttrs, getValue());
}

void pylir::Py::BoolAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                   llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::Attribute pylir::Py::BoolAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                 llvm::ArrayRef<mlir::Type>) const
{
    return doTypeObjectSlotsReplace(*this, replAttrs, getValue());
}

void pylir::Py::FloatAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                    llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::Attribute pylir::Py::FloatAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                  llvm::ArrayRef<mlir::Type>) const
{
    return doTypeObjectSlotsReplace(*this, replAttrs, getValue());
}

void pylir::Py::StrAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                  llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::Attribute pylir::Py::StrAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                llvm::ArrayRef<mlir::Type>) const
{
    return doTypeObjectSlotsReplace(*this, replAttrs, getValue());
}

void pylir::Py::TupleAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                    llvm::function_ref<void(mlir::Type)>) const
{
    for (auto& iter : getValue())
    {
        walkAttrsFn(iter);
    }
    walkAttrsFn(getTypeObject());
}

mlir::Attribute pylir::Py::TupleAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                  llvm::ArrayRef<mlir::Type>) const
{
    auto typeObject = replAttrs.back().cast<mlir::FlatSymbolRefAttr>();
    return get(getContext(), replAttrs.drop_back(), typeObject);
}

void pylir::Py::ListAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                   llvm::function_ref<void(mlir::Type)>) const
{
    for (auto& iter : getValue())
    {
        walkAttrsFn(iter);
    }
    walkAttrsFn(getTypeObject());
    walkAttrsFn(getSlots());
}

mlir::Attribute pylir::Py::ListAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                 llvm::ArrayRef<mlir::Type>) const
{
    auto typeObject = replAttrs.take_back(2).front().cast<mlir::FlatSymbolRefAttr>();
    auto slots = replAttrs.back().cast<mlir::DictionaryAttr>();
    return get(getContext(), replAttrs.drop_back(2), typeObject, slots);
}

void pylir::Py::SetAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                  llvm::function_ref<void(mlir::Type)>) const
{
    for (auto& iter : getValue())
    {
        walkAttrsFn(iter);
    }
    walkAttrsFn(getTypeObject());
    walkAttrsFn(getSlots());
}

mlir::Attribute pylir::Py::SetAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                llvm::ArrayRef<mlir::Type>) const
{
    auto typeObject = replAttrs.take_back(2).front().cast<mlir::FlatSymbolRefAttr>();
    auto slots = replAttrs.back().cast<mlir::DictionaryAttr>();
    return get(getContext(), replAttrs.drop_back(2), typeObject, slots);
}

void pylir::Py::FunctionAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                       llvm::function_ref<void(mlir::Type)>) const
{
    walkAttrsFn(getValue());
    walkAttrsFn(getQualName());
    walkAttrsFn(getDefaults());
    walkAttrsFn(getKwDefaults());
    if (getDict())
    {
        walkAttrsFn(getDict());
    }
}

mlir::Attribute pylir::Py::FunctionAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                     llvm::ArrayRef<mlir::Type>) const
{
    auto value = replAttrs[0].cast<mlir::FlatSymbolRefAttr>();
    auto qualName = replAttrs[1];
    auto defaults = replAttrs[2];
    auto kwDefaults = replAttrs[4];
    auto dict = replAttrs.size() > 5 ? replAttrs[5] : getDict();
    return get(getContext(), value, qualName, defaults, kwDefaults, dict);
}

void pylir::Py::TypeAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                   llvm::function_ref<void(mlir::Type)>) const
{
    walkAttrsFn(getMroTuple());
    walkAttrsFn(getTypeObject());
    walkAttrsFn(getSlots());
}

mlir::Attribute pylir::Py::TypeAttr::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute> replAttrs,
                                                                 llvm::ArrayRef<mlir::Type>) const
{
    auto value = replAttrs[0];
    auto typeObject = replAttrs[1].cast<mlir::FlatSymbolRefAttr>();
    auto slots = replAttrs[2].cast<mlir::DictionaryAttr>();
    return get(getContext(), value, typeObject, slots);
}
