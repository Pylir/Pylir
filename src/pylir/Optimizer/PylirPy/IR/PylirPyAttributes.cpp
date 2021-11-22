#include "PylirPyAttributes.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/TypeSwitch.h>

#include "PylirPyDialect.hpp"

namespace pylir::Py::detail
{
struct IntAttrStorage : public mlir::AttributeStorage
{
    IntAttrStorage(BigInt value) : value(std::move(value)) {}

    using KeyTy = BigInt;

    bool operator==(const KeyTy& other) const
    {
        return value == other;
    }

    static ::llvm::hash_code hashKey(const KeyTy& key)
    {
        auto count = mp_sbin_size(&key.getHandle());
        llvm::SmallVector<std::uint8_t, 10> data(count);
        auto result = mp_to_sbin(&key.getHandle(), data.data(), count, nullptr);
        PYLIR_ASSERT(result == MP_OKAY);
        return llvm::hash_value(makeArrayRef(data));
    }

    static IntAttrStorage* construct(::mlir::AttributeStorageAllocator& allocator, const KeyTy& key)
    {
        return new (allocator.allocate<IntAttrStorage>()) IntAttrStorage(key);
    }

    BigInt value;
};
} // namespace pylir::Py::detail

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.cpp.inc"

void pylir::Py::PylirPyDialect::initializeAttributes()
{
    addAttributes<BoolAttr,
#define GET_ATTRDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.cpp.inc"
                  >();
}

mlir::Attribute pylir::Py::PylirPyDialect::parseAttribute(::mlir::DialectAsmParser& parser, ::mlir::Type type) const
{
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword))
    {
        return {};
    }
    if (keyword == Py::BoolAttr::getMnemonic())
    {
        return Py::BoolAttr::parse(parser, type);
    }
    mlir::Attribute result;
    (void)generatedAttributeParser(parser, keyword, type, result);
    return result;
}

void pylir::Py::PylirPyDialect::printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter& os) const
{
    if (auto boolean = attr.dyn_cast_or_null<BoolAttr>())
    {
        os << boolean.getMnemonic();
        return boolean.print(os);
    }
    (void)generatedAttributePrinter(attr, os);
}

pylir::BigInt pylir::Py::IntAttr::getValue() const
{
    return getImpl()->value;
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

pylir::Py::BoolAttr pylir::Py::BoolAttr::get(::mlir::MLIRContext* context, bool value)
{
    return Base::get(context, BigInt(value ? 1 : 0));
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

bool pylir::Py::BoolAttr::getValue() const
{
    return !getImpl()->value.isZero();
}

void pylir::Py::ObjectAttr::print(::mlir::AsmPrinter& printer) const
{
    printer << "<type: " << getType();
    if (getBuiltinValue())
    {
        printer << ", value: " << *getBuiltinValue();
    }
    printer << ", __dict__: " << getAttributes() << ">";
}

mlir::Attribute pylir::Py::ObjectAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    mlir::Attribute type;
    Py::DictAttr dictAttr;
    if (parser.parseLess() || parser.parseKeyword("type") || parser.parseColon() || parser.parseAttribute(type)
        || parser.parseComma())
    {
        return {};
    }
    llvm::Optional<mlir::Attribute> builtinValue;
    if (!parser.parseOptionalKeyword("value"))
    {
        builtinValue.emplace();
        if (parser.parseColon() || parser.parseAttribute(*builtinValue) || parser.parseComma())
        {
            return {};
        }
    }
    if (parser.parseKeyword("__dict__") || parser.parseColon() || parser.parseAttribute(dictAttr)
        || parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), type, dictAttr, builtinValue);
}

void pylir::Py::ListAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                   llvm::function_ref<void(mlir::Type)>) const
{
    std::for_each(getValue().begin(), getValue().end(), walkAttrsFn);
}

mlir::SubElementAttrInterface pylir::Py::ListAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    auto vector = getValue().vec();
    for (auto [index, attr] : replacements)
    {
        vector[index] = attr;
    }
    return get(getContext(), vector);
}

void pylir::Py::TupleAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                    llvm::function_ref<void(mlir::Type)>) const
{
    std::for_each(getValue().begin(), getValue().end(), walkAttrsFn);
}

mlir::SubElementAttrInterface pylir::Py::TupleAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    auto vector = getValue().vec();
    for (auto [index, attr] : replacements)
    {
        vector[index] = attr;
    }
    return get(getContext(), vector);
}

::pylir::Py::SetAttr pylir::Py::SetAttr::get(::mlir::MLIRContext* context, llvm::ArrayRef<mlir::Attribute> attributes)
{
    // TODO: The order of a set iteration might be undefined, but it ought to still match up at compile time with
    //       runtime probably
    auto vector = attributes.vec();
    vector.erase(std::unique(vector.begin(), vector.end()), vector.end());
    return getUniqued(context, vector);
}

::pylir::Py::SetAttr pylir::Py::SetAttr::getUniqued(::mlir::MLIRContext* context,
                                                    llvm::ArrayRef<mlir::Attribute> attributes)
{
    return Base::get(context, attributes);
}

void pylir::Py::SetAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                  llvm::function_ref<void(mlir::Type)>) const
{
    std::for_each(getValue().begin(), getValue().end(), walkAttrsFn);
}

mlir::SubElementAttrInterface pylir::Py::SetAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    auto vector = getValue().vec();
    for (auto [index, attr] : replacements)
    {
        vector[index] = attr;
    }
    return get(getContext(), vector);
}

pylir::Py::DictAttr pylir::Py::DictAttr::get(::mlir::MLIRContext* context,
                                             llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> attributes)
{
    auto vector = attributes.vec();
    vector.erase(std::unique(vector.begin(), vector.end(),
                             [](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first; }),
                 vector.end());
    return getUniqued(context, vector);
}

pylir::Py::DictAttr
    pylir::Py::DictAttr::getUniqued(::mlir::MLIRContext* context,
                                    llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> attributes)
{
    return Base::get(context, attributes);
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
}

mlir::SubElementAttrInterface pylir::Py::DictAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    bool changedOrder = false;
    auto vector = getValue().vec();
    for (auto [index, attr] : replacements)
    {
        if (index & 1)
        {
            vector[index / 2].second = attr;
        }
        else
        {
            changedOrder = true;
            vector[index / 2].first = attr;
        }
    }
    if (changedOrder)
    {
        return get(getContext(), vector);
    }
    return getUniqued(getContext(), vector);
}

void pylir::Py::ObjectAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                     llvm::function_ref<void(mlir::Type)>) const
{
    walkAttrsFn(getType());
    walkAttrsFn(getAttributes());
    if (getBuiltinValue())
    {
        walkAttrsFn(*getBuiltinValue());
    }
}

mlir::SubElementAttrInterface pylir::Py::ObjectAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    auto type = getType();
    auto attributes = getAttributes();
    auto builtinValue = getBuiltinValue();
    for (auto [index, attr] : replacements)
    {
        switch (index)
        {
            case 0: type = attr; break;
            case 1: attributes = attr.cast<Py::DictAttr>(); break;
            case 2: builtinValue = attr; break;
            default: PYLIR_UNREACHABLE;
        }
    }
    return get(getContext(), type, attributes, builtinValue);
}
