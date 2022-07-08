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

mlir::SubElementAttrInterface pylir::Py::DictAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    auto type = getTypeObject();
    auto slots = getSlots();
    auto vector = getValue().vec();
    for (auto [index, attr] : replacements)
    {
        if (index == vector.size() * 2)
        {
            type = attr.cast<mlir::FlatSymbolRefAttr>();
        }
        else if (index == vector.size() * 2 + 1)
        {
            slots = attr.cast<mlir::DictionaryAttr>();
        }
        else if (index & 1)
        {
            vector[index / 2].second = attr;
        }
        else
        {
            vector[index / 2].first = attr;
        }
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
Op doTypeObjectSlotsReplace(Op op, ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements,
                            std::size_t offset, Args&&... prior)
{
    auto type = op.getTypeObject();
    auto slots = op.getSlots();
    for (auto [index, attr] : replacements)
    {
        if (index == offset)
        {
            type = attr.template cast<mlir::FlatSymbolRefAttr>();
        }
        else if (index == offset + 1)
        {
            slots = attr.template cast<mlir::DictionaryAttr>();
        }
    }
    return Op::get(op.getContext(), std::forward<Args>(prior)..., type, slots);
}
} // namespace

void pylir::Py::ObjectAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                     llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::SubElementAttrInterface pylir::Py::ObjectAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    return doTypeObjectSlotsReplace(*this, replacements, 0);
}

void pylir::Py::IntAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                  llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::SubElementAttrInterface pylir::Py::IntAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    return doTypeObjectSlotsReplace(*this, replacements, 0, getValue());
}

void pylir::Py::BoolAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                   llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::SubElementAttrInterface pylir::Py::BoolAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    return doTypeObjectSlotsReplace(*this, replacements, 0, getValue());
}

void pylir::Py::FloatAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                    llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::SubElementAttrInterface pylir::Py::FloatAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    return doTypeObjectSlotsReplace(*this, replacements, 0, getValue());
}

void pylir::Py::StrAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                  llvm::function_ref<void(mlir::Type)>) const
{
    doTypeObjectSlotsWalk(*this, walkAttrsFn);
}

mlir::SubElementAttrInterface pylir::Py::StrAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    return doTypeObjectSlotsReplace(*this, replacements, 0, getValue());
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

mlir::SubElementAttrInterface pylir::Py::TupleAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    mlir::FlatSymbolRefAttr typeObject = getTypeObject();
    auto vector = llvm::to_vector(getValue());
    for (auto& [index, attr] : replacements)
    {
        if (index == getValue().size())
        {
            typeObject = attr.cast<mlir::FlatSymbolRefAttr>();
        }
        else
        {
            vector[index] = attr;
        }
    }
    return get(getContext(), vector, typeObject);
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

mlir::SubElementAttrInterface pylir::Py::ListAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    mlir::FlatSymbolRefAttr typeObject = getTypeObject();
    auto slots = getSlots();
    auto vector = llvm::to_vector(getValue());
    for (auto& [index, attr] : replacements)
    {
        if (index == getValue().size())
        {
            typeObject = attr.cast<mlir::FlatSymbolRefAttr>();
        }
        else if (index == getValue().size() + 1)
        {
            slots = attr.cast<mlir::DictionaryAttr>();
        }
        else
        {
            vector[index] = attr;
        }
    }
    return get(getContext(), vector, typeObject, slots);
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

mlir::SubElementAttrInterface pylir::Py::SetAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    mlir::FlatSymbolRefAttr typeObject = getTypeObject();
    auto slots = getSlots();
    auto vector = llvm::to_vector(getValue());
    for (auto& [index, attr] : replacements)
    {
        if (index == getValue().size())
        {
            typeObject = attr.cast<mlir::FlatSymbolRefAttr>();
        }
        else if (index == getValue().size() + 1)
        {
            slots = attr.cast<mlir::DictionaryAttr>();
        }
        else
        {
            vector[index] = attr;
        }
    }
    return get(getContext(), vector, typeObject, slots);
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

mlir::SubElementAttrInterface pylir::Py::FunctionAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    auto value = getValue();
    auto qualName = getQualName();
    auto defaults = getDefaults();
    auto kwDefaults = getKwDefaults();
    auto dict = getDict();
    for (auto [index, attr] : replacements)
    {
        switch (index)
        {
            case 0: value = attr.cast<mlir::FlatSymbolRefAttr>(); break;
            case 1: qualName = attr; break;
            case 2: kwDefaults = attr; break;
            case 3: defaults = attr; break;
            case 4: dict = attr; break;
        }
    }
    return get(getContext(), value, qualName, defaults, kwDefaults, dict);
}

void pylir::Py::TypeAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                   llvm::function_ref<void(mlir::Type)>) const
{
    walkAttrsFn(getMroTuple());
    walkAttrsFn(getTypeObject());
    walkAttrsFn(getSlots());
}

mlir::SubElementAttrInterface pylir::Py::TypeAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    auto value = getMroTuple();
    auto typeObject = getTypeObject();
    auto slots = getSlots();
    for (auto [index, attr] : replacements)
    {
        switch (index)
        {
            case 0: value = attr; break;
            case 1: typeObject = attr.cast<mlir::FlatSymbolRefAttr>(); break;
            case 2: slots = attr.cast<mlir::DictionaryAttr>(); break;
        }
    }
    return get(getContext(), value, typeObject, slots);
}
