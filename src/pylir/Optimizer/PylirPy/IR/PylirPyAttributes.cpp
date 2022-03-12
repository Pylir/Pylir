#include "PylirPyAttributes.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <utility>

#include "PylirPyDialect.hpp"

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

void pylir::Py::IntAttr::print(::mlir::AsmPrinter& printer) const
{
    printer << "<" << getValue().toString() << ">";
}

mlir::Attribute pylir::Py::IntAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    llvm::APInt apInt;
    if (parser.parseLess() || parser.parseInteger(apInt) || parser.parseGreater())
    {
        return {};
    }
    llvm::SmallString<10> str;
    apInt.toStringSigned(str);
    return IntAttr::get(parser.getContext(), BigInt(std::string{str.data(), str.size()}));
}

void pylir::Py::ListAttr::print(::mlir::AsmPrinter& printer) const
{
    printer << "<[";
    llvm::interleaveComma(getValue(), printer);
    printer << "]>";
}

mlir::Attribute pylir::Py::ListAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    llvm::SmallVector<mlir::Attribute> attrs;
    if (parser.parseLess()
        || parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square,
                                          [&] { return parser.parseAttribute(attrs.emplace_back()); })
        || parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), attrs);
}

void pylir::Py::TupleAttr::print(::mlir::AsmPrinter& printer) const
{
    printer << "<(";
    llvm::interleaveComma(getValue(), printer);
    printer << ")>";
}

mlir::Attribute pylir::Py::TupleAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    llvm::SmallVector<mlir::Attribute> attrs;
    if (parser.parseLess()
        || parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Paren,
                                          [&] { return parser.parseAttribute(attrs.emplace_back()); })
        || parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), attrs);
}

void pylir::Py::DictAttr::print(::mlir::AsmPrinter& printer) const
{
    printer << "<{";
    llvm::interleaveComma(getValue(), printer, [&](auto&& pair) { printer << pair.first << " to " << pair.second; });
    printer << "}>";
}

mlir::Attribute pylir::Py::DictAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    llvm::SmallVector<std::pair<mlir::Attribute, mlir::Attribute>> attrs;
    if (parser.parseLess()
        || parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Braces,
                                          [&]
                                          {
                                              return mlir::failure(parser.parseAttribute(attrs.emplace_back().first)
                                                                   || parser.parseKeyword("to")
                                                                   || parser.parseAttribute(attrs.back().second));
                                          })
        || parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), attrs);
}

mlir::Attribute pylir::Py::BoolAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    if (parser.parseLess())
    {
        return {};
    }
    llvm::StringRef keyword;
    auto loc = parser.getCurrentLocation();
    if (parser.parseKeyword(&keyword))
    {
        return {};
    }
    if (keyword != "True" && keyword != "False")
    {
        parser.emitError(loc, "Expected 'True' or 'False' instead of ") << keyword;
        return {};
    }
    if (parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), keyword == "True");
}

void pylir::Py::BoolAttr::print(::mlir::AsmPrinter& printer) const
{
    printer << "<" << (getValue() ? "True" : "False") << ">";
}

mlir::Attribute pylir::Py::FloatAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    double value;
    if (parser.parseLess() || parser.parseFloat(value) || parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), llvm::APFloat(value));
}

void pylir::Py::FloatAttr::print(::mlir::AsmPrinter& printer) const
{
    printer << "<" << getValue() << ">";
}

mlir::Attribute pylir::Py::StrAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    std::string value;
    if (parser.parseLess() || parser.parseString(&value) || parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), value);
}

void pylir::Py::StrAttr::print(::mlir::AsmPrinter& printer) const
{
    printer << "<\"";
    llvm::printEscapedString(getValue(), printer.getStream());
    printer << "\">";
}

void pylir::Py::SetAttr::print(::mlir::AsmPrinter& printer) const
{
    printer << "<{";
    llvm::interleaveComma(getValue(), printer);
    printer << "}>";
}

mlir::Attribute pylir::Py::SetAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    llvm::SmallVector<mlir::Attribute> attrs;
    if (parser.parseLess()
        || parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Braces,
                                          [&] { return parser.parseAttribute(attrs.emplace_back()); })
        || parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), attrs);
}

mlir::Attribute pylir::Py::FunctionAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    mlir::FlatSymbolRefAttr symbol;
    if (parser.parseLess() || parser.parseAttribute(symbol))
    {
        return {};
    }
    mlir::Attribute defaults;
    mlir::Attribute kwDefaults;
    mlir::Attribute dict;
    mlir::Attribute qualName;
    while (!parser.parseOptionalComma())
    {
        mlir::Attribute attribute;
        llvm::StringRef keyword;
        auto loc = parser.getCurrentLocation();
        if (parser.parseKeyword(&keyword) || parser.parseColon() || parser.parseAttribute(attribute))
        {
            return {};
        }
        if (keyword == "__defaults__")
        {
            defaults = attribute;
        }
        else if (keyword == "__kwdefaults__")
        {
            kwDefaults = attribute;
        }
        else if (keyword == "__dict__")
        {
            dict = attribute;
        }
        else if (keyword == "__qualname__")
        {
            qualName = attribute;
        }
        else
        {
            parser.emitError(loc, "Invalid keyword '") << keyword << "'";
            return {};
        }
    }
    if (parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), symbol, qualName, defaults, kwDefaults, dict);
}

void pylir::Py::FunctionAttr::print(::mlir::AsmPrinter& printer) const
{
    printer << "<" << getValue();
    if (auto qualName = getQualName(); !qualName.isa<Py::StrAttr>() || !qualName.cast<Py::StrAttr>().getValue().empty())
    {
        printer << ", __qualname__: " << qualName;
    }
    if (auto defaults = getDefaults(); defaults != mlir::FlatSymbolRefAttr::get(getContext(), Builtins::None.name))
    {
        printer << ", __defaults__: " << defaults;
    }
    if (auto kwDefaults = getKwDefaults();
        kwDefaults != mlir::FlatSymbolRefAttr::get(getContext(), Builtins::None.name))
    {
        printer << ", __kwdefaults__: " << kwDefaults;
    }
    if (auto dict = getDict())
    {
        printer << ", __dict__: " << dict;
    }
    printer << ">";
}

mlir::Attribute pylir::Py::TypeAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    if (parser.parseOptionalLess())
    {
        return get(parser.getContext());
    }
    mlir::DictionaryAttr slots;
    mlir::Attribute mroTuple;
    llvm::SMLoc loc;
    auto action = [&](llvm::StringRef keyword) -> mlir::LogicalResult
    {
        if (keyword == "slots")
        {
            if (slots)
            {
                return parser.emitError(loc, "'slots' can only appear once");
            }
            return parser.parseAttribute(slots);
        }
        if (keyword == "mro")
        {
            if (mroTuple)
            {
                return parser.emitError(loc, "'slots' can only appear once");
            }
            return parser.parseAttribute(mroTuple);
        }
        return mlir::failure();
    };
    loc = parser.getCurrentLocation();
    llvm::StringRef result;
    if (parser.parseKeyword(&result) || parser.parseColon() || mlir::failed(action(result)))
    {
        return {};
    }
    while (!parser.parseOptionalComma())
    {
        loc = parser.getCurrentLocation();
        if (parser.parseKeyword(&result) || parser.parseColon() || mlir::failed(action(result)))
        {
            return {};
        }
    }
    if (parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), mroTuple, {}, slots);
}

void pylir::Py::TypeAttr::print(::mlir::AsmPrinter& printer) const
{
    auto slots = getSlots();
    auto mro = getMroTuple();
    bool mroDefault = mro.isa<pylir::Py::TupleAttr>() && mro.cast<pylir::Py::TupleAttr>().getValue().empty();
    if (slots.empty() && mroDefault)
    {
        return;
    }
    printer << "<";
    if (!slots.empty())
    {
        printer << "slots: " << slots;
        if (!mroDefault)
        {
            printer << ", mro: " << mro;
        }
    }
    else
    {
        printer << "mro: " << mro;
    }
    printer << ">";
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
            type = attr.cast<mlir::FlatSymbolRefAttr>();
        }
        else if (index == offset + 1)
        {
            slots = attr.cast<mlir::DictionaryAttr>();
        }
    }
    return Op::get(op.getContext(), std::forward<Args>(prior)..., type, slots);
}
} // namespace

void pylir::Py::ObjectAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                     llvm::function_ref<void(mlir::Type)>) const
{
    walkAttrsFn(getTypeObject());
    walkAttrsFn(getSlots());
    if (getBuiltinValue())
    {
        walkAttrsFn(getBuiltinValue());
    }
}

mlir::SubElementAttrInterface pylir::Py::ObjectAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    auto type = getTypeObject();
    auto slots = getSlots();
    auto builtinValue = getBuiltinValue();
    for (auto [index, attr] : replacements)
    {
        switch (index)
        {
            case 0: type = attr.cast<mlir::FlatSymbolRefAttr>(); break;
            case 1: slots = attr.cast<mlir::DictionaryAttr>(); break;
            case 2: builtinValue = attr; break;
            default: PYLIR_UNREACHABLE;
        }
    }
    return get(type, slots, builtinValue);
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
    auto kwDefaults = getKwDefaults();
    auto dict = getDict();
    for (auto [index, attr] : replacements)
    {
        switch (index)
        {
            case 0: value = attr.cast<mlir::FlatSymbolRefAttr>(); break;
            case 1: qualName = attr; break;
            case 2: kwDefaults = attr; break;
            case 3: dict = attr; break;
        }
    }
    return get(getContext(), value, qualName, kwDefaults, dict);
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
