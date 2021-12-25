#include "PylirPyAttributes.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include "PylirPyDialect.hpp"

namespace pylir::Py::detail
{
struct IntImplAttrStorage : public mlir::AttributeStorage
{
    IntImplAttrStorage(BigInt value) : value(std::move(value)) {}

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

    static IntImplAttrStorage* construct(::mlir::AttributeStorageAllocator& allocator, const KeyTy& key)
    {
        return new (allocator.allocate<IntImplAttrStorage>()) IntImplAttrStorage(key);
    }

    BigInt value;
};

struct ObjectAttrStorage : public ::mlir::AttributeStorage
{
    using KeyTy = std::tuple<mlir::FlatSymbolRefAttr, ::pylir::Py::SlotsAttr, mlir::Attribute>;
    ObjectAttrStorage(mlir::FlatSymbolRefAttr type, ::pylir::Py::SlotsAttr slots, mlir::Attribute builtinValue)
        : ::mlir::AttributeStorage(), type(type), slots(slots), builtinValue(builtinValue)
    {
    }

    bool operator==(const KeyTy& tblgenKey) const
    {
        return (type == std::get<0>(tblgenKey)) && (slots == std::get<1>(tblgenKey))
               && (builtinValue == std::get<2>(tblgenKey));
    }

    static ::llvm::hash_code hashKey(const KeyTy& tblgenKey)
    {
        return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
    }

    static ObjectAttrStorage* construct(::mlir::AttributeStorageAllocator& allocator, const KeyTy& tblgenKey)
    {
        auto type = std::get<0>(tblgenKey);
        auto slots = std::get<1>(tblgenKey);
        auto builtinValue = std::get<2>(tblgenKey);
        return new (allocator.allocate<ObjectAttrStorage>()) ObjectAttrStorage(type, slots, builtinValue);
    }

    mlir::FlatSymbolRefAttr type;
    ::pylir::Py::SlotsAttr slots;
    mlir::Attribute builtinValue;
};

} // namespace pylir::Py::detail

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.cpp.inc"

void pylir::Py::PylirPyDialect::initializeAttributes()
{
    addAttributes<ObjectAttr,
#define GET_ATTRDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.cpp.inc"
                  >();
}

mlir::Attribute pylir::Py::PylirPyDialect::parseAttribute(::mlir::DialectAsmParser& parser, ::mlir::Type type) const
{
    auto loc = parser.getCurrentLocation();
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword))
    {
        return {};
    }
    if (keyword == Py::IntAttr::getMnemonic())
    {
        return Py::IntAttr::parseMethod(parser, type);
    }
    if (keyword == Py::BoolAttr::getMnemonic())
    {
        return Py::BoolAttr::parseMethod(parser, type);
    }
    if (keyword == Py::FloatAttr::getMnemonic())
    {
        return Py::FloatAttr::parseMethod(parser, type);
    }
    if (keyword == Py::StringAttr::getMnemonic())
    {
        return Py::StringAttr::parseMethod(parser, type);
    }
    if (keyword == Py::TupleAttr::getMnemonic())
    {
        return Py::TupleAttr::parseMethod(parser, type);
    }
    if (keyword == Py::ListAttr::getMnemonic())
    {
        return Py::ListAttr::parseMethod(parser, type);
    }
    if (keyword == Py::SetAttr::getMnemonic())
    {
        return Py::SetAttr::parseMethod(parser, type);
    }
    if (keyword == Py::DictAttr::getMnemonic())
    {
        return Py::DictAttr::parseMethod(parser, type);
    }
    if (keyword == Py::FunctionAttr::getMnemonic())
    {
        return Py::FunctionAttr::parseMethod(parser, type);
    }
    if (keyword == Py::TypeAttr::getMnemonic())
    {
        return Py::TypeAttr::parseMethod(parser, type);
    }
    if (keyword == Py::ObjectAttr::getMnemonic())
    {
        return Py::ObjectAttr::parseMethod(parser, type);
    }
    mlir::Attribute result;
    if (!generatedAttributeParser(parser, keyword, type, result).hasValue())
    {
        parser.emitError(loc) << "unknown  type `" << keyword << "` in dialect `" << getNamespace() << "`";
    }
    return result;
}

void pylir::Py::PylirPyDialect::printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter& os) const
{
    llvm::TypeSwitch<mlir::Attribute>(attr)
        .Case<BoolAttr, IntAttr, FloatAttr, StringAttr, TupleAttr, ListAttr, SetAttr, DictAttr, FunctionAttr, TypeAttr,
              ObjectAttr>(
            [&](auto attr)
            {
                os << attr.getMnemonic();
                attr.printMethod(os);
            })
        .Default([&](auto attr) { (void)generatedAttributePrinter(attr, os); });
}

pylir::Py::IntAttr pylir::Py::IntAttr::get(::mlir::MLIRContext* context, BigInt value)
{
    return ObjectAttr::get(mlir::FlatSymbolRefAttr::get(context, Builtins::Int.name), {},
                           IntImplAttr::get(context, value))
        .cast<IntAttr>();
}

pylir::BigInt pylir::Py::IntAttr::getValue() const
{
    return this->getBuiltinValue().cast<IntImplAttr>().getValue();
}

void pylir::Py::IntAttr::printMethod(::mlir::AsmPrinter& printer) const
{
    printer << "<" << getValue().toString() << ">";
}

mlir::Attribute pylir::Py::IntAttr::parseMethod(::mlir::AsmParser& parser, ::mlir::Type)
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

pylir::Py::ListAttr pylir::Py::ListAttr::get(::mlir::MLIRContext* context, llvm::ArrayRef<mlir::Attribute> value)
{
    return ObjectAttr::get(mlir::FlatSymbolRefAttr::get(context, Builtins::List.name), {},
                           mlir::ArrayAttr::get(context, value))
        .cast<ListAttr>();
}

mlir::ArrayAttr pylir::Py::ListAttr::getValueAttr() const
{
    return getBuiltinValue().cast<mlir::ArrayAttr>();
}

llvm::ArrayRef<mlir::Attribute> pylir::Py::ListAttr::getValue() const
{
    return getValueAttr().getValue();
}

void pylir::Py::ListAttr::printMethod(::mlir::AsmPrinter& printer) const
{
    printer << "<[";
    llvm::interleaveComma(getValue(), printer);
    printer << "]>";
}

mlir::Attribute pylir::Py::ListAttr::parseMethod(::mlir::AsmParser& parser, ::mlir::Type)
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

pylir::Py::TupleAttr pylir::Py::TupleAttr::get(::mlir::MLIRContext* context, llvm::ArrayRef<mlir::Attribute> value)
{
    return ObjectAttr::get(mlir::FlatSymbolRefAttr::get(context, Builtins::Tuple.name), {},
                           mlir::ArrayAttr::get(context, value))
        .cast<TupleAttr>();
}

mlir::ArrayAttr pylir::Py::TupleAttr::getValueAttr() const
{
    return getBuiltinValue().cast<mlir::ArrayAttr>();
}

llvm::ArrayRef<mlir::Attribute> pylir::Py::TupleAttr::getValue() const
{
    return getValueAttr().getValue();
}

void pylir::Py::TupleAttr::printMethod(::mlir::AsmPrinter& printer) const
{
    printer << "<(";
    llvm::interleaveComma(getValue(), printer);
    printer << ")>";
}

mlir::Attribute pylir::Py::TupleAttr::parseMethod(::mlir::AsmParser& parser, ::mlir::Type)
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

pylir::Py::DictAttr pylir::Py::DictAttr::get(::mlir::MLIRContext* context,
                                             llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value)
{
    return ObjectAttr::get(mlir::FlatSymbolRefAttr::get(context, Builtins::Dict.name), {},
                           Py::DictImplAttr::get(context, value))
        .cast<DictAttr>();
}

pylir::Py::DictAttr pylir::Py::DictAttr::getUniqued(::mlir::MLIRContext* context,
                                                    llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value)
{
    return ObjectAttr::get(mlir::FlatSymbolRefAttr::get(context, Builtins::Dict.name), {},
                           Py::DictImplAttr::getUniqued(context, value))
        .cast<DictAttr>();
}

llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> pylir::Py::DictAttr::getValue() const
{
    return getBuiltinValue().cast<DictImplAttr>().getValue();
}

void pylir::Py::DictAttr::printMethod(::mlir::AsmPrinter& printer) const
{
    printer << "<{";
    llvm::interleaveComma(getValue(), printer, [&](auto&& pair) { printer << pair.first << " to " << pair.second; });
    printer << "}>";
}

mlir::Attribute pylir::Py::DictAttr::parseMethod(::mlir::AsmParser& parser, ::mlir::Type)
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
    return ObjectAttr::get(mlir::FlatSymbolRefAttr::get(context, Builtins::Bool.name), {},
                           IntImplAttr::get(context, BigInt(value ? 1 : 0)))
        .cast<BoolAttr>();
}

mlir::Attribute pylir::Py::BoolAttr::parseMethod(::mlir::AsmParser& parser, ::mlir::Type)
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

void pylir::Py::BoolAttr::printMethod(::mlir::AsmPrinter& printer) const
{
    printer << "<" << (getValue() ? "True" : "False") << ">";
}

bool pylir::Py::BoolAttr::getValue() const
{
    return !this->cast<IntAttr>().getValue().isZero();
}

pylir::Py::FloatAttr pylir::Py::FloatAttr::get(::mlir::MLIRContext* context, double value)
{
    return ObjectAttr::get(mlir::FlatSymbolRefAttr::get(context, Builtins::Float.name), {},
                           mlir::FloatAttr::get(mlir::Float64Type::get(context), value))
        .cast<FloatAttr>();
}

mlir::FloatAttr pylir::Py::FloatAttr::getValueAttr() const
{
    return getBuiltinValue().cast<mlir::FloatAttr>();
}

double pylir::Py::FloatAttr::getValue() const
{
    return getValueAttr().getValueAsDouble();
}

mlir::Attribute pylir::Py::FloatAttr::parseMethod(::mlir::AsmParser& parser, ::mlir::Type)
{
    double value;
    if (parser.parseLess() || parser.parseFloat(value) || parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), value);
}

void pylir::Py::FloatAttr::printMethod(::mlir::AsmPrinter& printer) const
{
    printer << "<" << getValue() << ">";
}

pylir::Py::StringAttr pylir::Py::StringAttr::get(::mlir::MLIRContext* context, llvm::StringRef value)
{
    return ObjectAttr::get(mlir::FlatSymbolRefAttr::get(context, Builtins::Str.name), {},
                           mlir::StringAttr::get(context, value))
        .cast<StringAttr>();
}

mlir::StringAttr pylir::Py::StringAttr::getValueAttr() const
{
    return getBuiltinValue().cast<mlir::StringAttr>();
}

llvm::StringRef pylir::Py::StringAttr::getValue() const
{
    return getValueAttr().getValue();
}

mlir::Attribute pylir::Py::StringAttr::parseMethod(::mlir::AsmParser& parser, ::mlir::Type)
{
    std::string value;
    if (parser.parseLess() || parser.parseString(&value) || parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), value);
}

void pylir::Py::StringAttr::printMethod(::mlir::AsmPrinter& printer) const
{
    printer << "<\"";
    llvm::printEscapedString(getValue(), printer.getStream());
    printer << "\">";
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
    return ObjectAttr::get(mlir::FlatSymbolRefAttr::get(context, Builtins::Set.name), {},
                           mlir::ArrayAttr::get(context, attributes))
        .cast<SetAttr>();
}

mlir::ArrayAttr pylir::Py::SetAttr::getValueAttr() const
{
    return getBuiltinValue().cast<mlir::ArrayAttr>();
}

llvm::ArrayRef<mlir::Attribute> pylir::Py::SetAttr::getValue() const
{
    return getValueAttr().getValue();
}

void pylir::Py::SetAttr::printMethod(::mlir::AsmPrinter& printer) const
{
    printer << "<{";
    llvm::interleaveComma(getValue(), printer);
    printer << "}>";
}

mlir::Attribute pylir::Py::SetAttr::parseMethod(::mlir::AsmParser& parser, ::mlir::Type)
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

pylir::Py::FunctionAttr pylir::Py::FunctionAttr::get(mlir::FlatSymbolRefAttr value, mlir::Attribute defaults,
                                                     mlir::Attribute kwDefaults, mlir::Attribute dict)
{
    if (!defaults)
    {
        defaults = mlir::SymbolRefAttr::get(value.getContext(), Builtins::None.name);
    }
    if (!kwDefaults)
    {
        kwDefaults = mlir::SymbolRefAttr::get(value.getContext(), Builtins::None.name);
    }
    llvm::SmallVector<std::pair<mlir::StringAttr, mlir::Attribute>> slots = {
        {mlir::StringAttr::get(value.getContext(), "__defaults__"), defaults},
        {mlir::StringAttr::get(value.getContext(), "__kwdefaults__"), kwDefaults}};
    if (dict)
    {
        slots.emplace_back(mlir::StringAttr::get(value.getContext(), "__dict__"), dict);
    }
    return ObjectAttr::get(mlir::FlatSymbolRefAttr::get(value.getContext(), Builtins::Function.name),
                           Py::SlotsAttr::get(value.getContext(), slots), value)
        .cast<FunctionAttr>();
}

mlir::Attribute pylir::Py::FunctionAttr::parseMethod(::mlir::AsmParser& parser, ::mlir::Type)
{
    mlir::FlatSymbolRefAttr symbol;
    if (parser.parseLess() || parser.parseAttribute(symbol))
    {
        return {};
    }
    mlir::Attribute defaults;
    mlir::Attribute kwDefaults;
    mlir::Attribute dict;
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
    return get(symbol, defaults, kwDefaults, dict);
}

void pylir::Py::FunctionAttr::printMethod(::mlir::AsmPrinter& printer) const
{
    printer << "<" << getValue();
    if (auto defaults = getDefaults(); defaults != mlir::FlatSymbolRefAttr::get(getContext(), Builtins::None.name))
    {
        printer << ", __defaults__: " << defaults;
    }
    if (auto kwDefaults = getKWDefaults();
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

mlir::FlatSymbolRefAttr pylir::Py::FunctionAttr::getValue() const
{
    return getBuiltinValue().cast<mlir::FlatSymbolRefAttr>();
}

mlir::Attribute pylir::Py::FunctionAttr::getDefaults() const
{
    return std::find_if(getSlots().getValue().begin(), getSlots().getValue().end(),
                        [](auto pair) { return pair.first.getValue() == "__defaults__"; })
        ->second;
}

mlir::Attribute pylir::Py::FunctionAttr::getKWDefaults() const
{
    return std::find_if(getSlots().getValue().begin(), getSlots().getValue().end(),
                        [](auto pair) { return pair.first.getValue() == "__kwdefaults__"; })
        ->second;
}

mlir::Attribute pylir::Py::FunctionAttr::getDict() const
{
    auto result = std::find_if(getSlots().getValue().begin(), getSlots().getValue().end(),
                               [](auto pair) { return pair.first.getValue() == "__dict__"; });
    if (result == getSlots().getValue().end())
    {
        return {};
    }
    return result->second;
}

pylir::Py::TypeAttr pylir::Py::TypeAttr::get(mlir::MLIRContext* context, ::pylir::Py::SlotsAttr slots)
{
    return ObjectAttr::get(mlir::FlatSymbolRefAttr::get(context, Builtins::Type.name), slots).cast<TypeAttr>();
}

mlir::Attribute pylir::Py::TypeAttr::parseMethod(::mlir::AsmParser& parser, ::mlir::Type)
{
    if (parser.parseOptionalLess())
    {
        return get(parser.getContext());
    }
    Py::SlotsAttr slots;
    if (parser.parseKeyword("slots") || parser.parseColon() || parser.parseAttribute(slots) || parser.parseGreater())
    {
        return {};
    }
    return get(parser.getContext(), slots);
}

void pylir::Py::TypeAttr::printMethod(::mlir::AsmPrinter& printer) const
{
    if (getSlots().getValue().empty())
    {
        return;
    }
    printer << "<slots: " << getSlots() << ">";
}

pylir::BigInt pylir::Py::IntImplAttr::getValue() const
{
    return getImpl()->value;
}

pylir::Py::DictImplAttr
    pylir::Py::DictImplAttr::get(::mlir::MLIRContext* context,
                                 llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> attributes)
{
    auto vector = attributes.vec();
    vector.erase(std::unique(vector.begin(), vector.end(),
                             [](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first; }),
                 vector.end());
    return getUniqued(context, vector);
}

pylir::Py::DictImplAttr
    pylir::Py::DictImplAttr::getUniqued(::mlir::MLIRContext* context,
                                        llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> attributes)
{
    return Base::get(context, attributes);
}

void pylir::Py::DictImplAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                       llvm::function_ref<void(mlir::Type)>) const
{
    std::for_each(getValue().begin(), getValue().end(),
                  [&](auto&& pair)
                  {
                      walkAttrsFn(pair.first);
                      walkAttrsFn(pair.second);
                  });
}

mlir::SubElementAttrInterface pylir::Py::DictImplAttr::replaceImmediateSubAttribute(
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

void pylir::Py::SlotsAttr::print(::mlir::AsmPrinter& printer) const
{
    printer << "<{";
    llvm::interleaveComma(getValue(), printer.getStream(),
                          [&](auto pair) { printer << pair.first << " to " << pair.second; });
    printer << "}>";
}

mlir::Attribute pylir::Py::SlotsAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    llvm::SmallVector<std::pair<mlir::StringAttr, mlir::Attribute>> attrs;
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

void pylir::Py::SlotsAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                    llvm::function_ref<void(mlir::Type)>) const
{
    for (auto [str, attr] : getValue())
    {
        walkAttrsFn(str);
        walkAttrsFn(attr);
    }
}

mlir::SubElementAttrInterface pylir::Py::SlotsAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    auto vector = getValue().vec();
    for (auto [index, attr] : replacements)
    {
        if (index & 1)
        {
            vector[index / 2].second = attr;
        }
        else
        {
            vector[index / 2].first = attr.cast<mlir::StringAttr>();
        }
    }
    return get(getContext(), vector);
}

pylir::Py::ObjectAttr pylir::Py::ObjectAttr::get(mlir::FlatSymbolRefAttr type)
{
    return get(type, {}, {});
}

pylir::Py::ObjectAttr pylir::Py::ObjectAttr::get(mlir::FlatSymbolRefAttr type, pylir::Py::SlotsAttr slots,
                                                 mlir::Attribute builtinValue)
{
    if (!slots)
    {
        slots = Py::SlotsAttr::get(type.getContext(), {});
    }
    return Base::get(type.getContext(), type, slots, builtinValue);
}

mlir::FlatSymbolRefAttr pylir::Py::ObjectAttr::getType() const
{
    return getImpl()->type;
}

::pylir::Py::SlotsAttr pylir::Py::ObjectAttr::getSlots() const
{
    return getImpl()->slots;
}

mlir::Attribute pylir::Py::ObjectAttr::getBuiltinValue() const
{
    return getImpl()->builtinValue;
}

void pylir::Py::ObjectAttr::printMethod(::mlir::AsmPrinter& printer) const
{
    printer << "<type: " << getType();
    if (!getSlots().getValue().empty())
    {
        printer << ", slots: " << getSlots();
    }
    if (getBuiltinValue())
    {
        printer << ", value: " << getBuiltinValue();
    }
    printer << ">";
}

mlir::Attribute pylir::Py::ObjectAttr::parseMethod(::mlir::AsmParser& parser, ::mlir::Type)
{
    mlir::FlatSymbolRefAttr type;
    if (parser.parseLess() || parser.parseKeyword("type") || parser.parseColon() || parser.parseAttribute(type))
    {
        return {};
    }
    pylir::Py::SlotsAttr slots;
    mlir::Attribute builtinValue;
    while (!parser.parseOptionalComma())
    {
        llvm::StringRef keyword;
        auto loc = parser.getCurrentLocation();
        if (parser.parseKeyword(&keyword))
        {
            return {};
        }
        if (keyword == "slots")
        {
            if (slots)
            {
                parser.emitError(loc, "`slots` can only appear once");
                return {};
            }
            if (parser.parseColon() || parser.parseAttribute(slots))
            {
                return {};
            }
        }
        else if (keyword == "value")
        {
            if (builtinValue)
            {
                parser.emitError(loc, "`value` can only appear once");
                return {};
            }
            if (parser.parseColon() || parser.parseAttribute(builtinValue))
            {
                return {};
            }
        }
        else
        {
            parser.emitError(loc, "Unexpected keyword `") << keyword << "`. Expected one of `slots` or `value`";
            return {};
        }
    }
    if (parser.parseGreater())
    {
        return {};
    }
    if (!slots)
    {
        slots = Py::SlotsAttr::get(parser.getContext(), {});
    }
    return get(type, slots, builtinValue);
}

void pylir::Py::ObjectAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                     llvm::function_ref<void(mlir::Type)>) const
{
    walkAttrsFn(getType());
    walkAttrsFn(getSlots());
    if (getBuiltinValue())
    {
        walkAttrsFn(getBuiltinValue());
    }
}

mlir::SubElementAttrInterface pylir::Py::ObjectAttr::replaceImmediateSubAttribute(
    ::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const
{
    auto type = getType();
    auto slots = getSlots();
    auto builtinValue = getBuiltinValue();
    for (auto [index, attr] : replacements)
    {
        switch (index)
        {
            case 0: type = attr.cast<mlir::FlatSymbolRefAttr>(); break;
            case 1: slots = attr.cast<Py::SlotsAttr>(); break;
            case 2: builtinValue = attr; break;
            default: PYLIR_UNREACHABLE;
        }
    }
    return get(type, slots, builtinValue);
}
