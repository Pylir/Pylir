#include "PylirDialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Functional.hpp>

#include "PylirAttributes.hpp"
#include "PylirOps.hpp"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/Dialect/PylirOpsTypes.cpp.inc"

void pylir::Dialect::PylirDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "pylir/Optimizer/Dialect/PylirOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Optimizer/Dialect/PylirOpsTypes.cpp.inc"
        >();
    addAttributes<NoneAttr, BoolAttr, FloatAttr, IntegerAttr, StringAttr, ListAttr, TupleAttr, SetAttr, DictAttr,
                  NotImplementedAttr>();
}

mlir::Type pylir::Dialect::PylirDialect::parseType(::mlir::DialectAsmParser& parser) const
{
    llvm::StringRef ref;
    if (parser.parseKeyword(&ref))
    {
        return {};
    }
    return generatedTypeParser(getContext(), parser, ref);
}

void pylir::Dialect::PylirDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter& os) const
{
    generatedTypePrinter(type, os);
}

mlir::Attribute pylir::Dialect::PylirDialect::parseAttribute(mlir::DialectAsmParser& parser, mlir::Type) const
{
    llvm::StringRef ref;
    if (parser.parseKeyword(&ref))
    {
        return {};
    }
    if (ref == "none")
    {
        return NoneAttr::get(getContext());
    }
    if (ref == "notImplemented")
    {
        return NotImplementedAttr::get(getContext());
    }
    if (ref == "float")
    {
        double value;
        if (parser.parseLess() || parser.parseFloat(value) || parser.parseGreater())
        {
            return {};
        }
        return FloatAttr::get(getContext(), value);
    }
    if (ref == "bool")
    {
        llvm::StringRef boolValue;
        if (parser.parseLess())
        {
            return {};
        }
        auto loc = parser.getCurrentLocation();
        if (parser.parseKeyword(&boolValue) || parser.parseGreater())
        {
            return {};
        }
        if (boolValue != "true" && boolValue != "false")
        {
            parser.emitError(loc, "Expected 'true' or 'false'");
            return {};
        }
        return BoolAttr::get(getContext(), boolValue == "true");
    }
    if (ref == "integer")
    {
        llvm::StringRef value;
        if (parser.parseLess())
        {
            return {};
        }
        auto loc = parser.getCurrentLocation();
        if (parser.parseOptionalString(&value))
        {
            parser.emitError(loc, "Expected string literal");
            return {};
        }
        if (parser.parseGreater())
        {
            return {};
        }
        llvm::APInt integer;
        if (value.getAsInteger(10, integer))
        {
            parser.emitError(loc, "Expected valid integer in string literal");
            return {};
        }
        return IntegerAttr::get(getContext(), std::move(integer));
    }
    if (ref == "string")
    {
        llvm::StringRef value;
        if (parser.parseLess())
        {
            return {};
        }
        auto loc = parser.getCurrentLocation();
        if (parser.parseOptionalString(&value))
        {
            parser.emitError(loc, "Expected string literal");
            return {};
        }
        if (parser.parseGreater())
        {
            return {};
        }
        return StringAttr::get(getContext(), value.str());
    }
    if (ref == "list")
    {
        if (parser.parseLess() || parser.parseLSquare())
        {
            return {};
        }
        std::vector<mlir::Attribute> attributes;
        {
            mlir::Attribute attribute;
            if (parser.parseAttribute(attribute))
            {
                return {};
            }
            attributes.push_back(attribute);
        }
        while (!parser.parseOptionalComma())
        {
            mlir::Attribute attribute;
            if (parser.parseAttribute(attribute))
            {
                return {};
            }
            attributes.push_back(attribute);
        }
        if (parser.parseRSquare() || parser.parseGreater())
        {
            return {};
        }
        return ListAttr::get(getContext(), attributes);
    }
    if (ref == "set")
    {
        if (parser.parseLess() || parser.parseLBrace())
        {
            return {};
        }
        llvm::DenseSet<mlir::Attribute> attributes;
        {
            mlir::Attribute attribute;
            if (parser.parseAttribute(attribute))
            {
                return {};
            }
            attributes.insert(attribute);
        }
        while (!parser.parseOptionalComma())
        {
            mlir::Attribute attribute;
            if (parser.parseAttribute(attribute))
            {
                return {};
            }
            attributes.insert(attribute);
        }
        if (parser.parseRBrace() || parser.parseGreater())
        {
            return {};
        }
        return SetAttr::get(getContext(), attributes);
    }
    if (ref == "dict")
    {
        if (parser.parseLess() || parser.parseLBrace())
        {
            return {};
        }
        llvm::DenseMap<mlir::Attribute, mlir::Attribute> map;
        std::vector<std::pair<mlir::Attribute, mlir::Attribute>> values;
        {
            mlir::Attribute key, value;
            if (parser.parseAttribute(key) || parser.parseColon() || parser.parseAttribute(value))
            {
                return {};
            }
            if (map.insert({key, value}).second)
            {
                values.emplace_back(key, value);
            }
        }
        while (!parser.parseOptionalComma())
        {
            mlir::Attribute key, value;
            if (parser.parseAttribute(key) || parser.parseColon() || parser.parseAttribute(value))
            {
                return {};
            }
            if (map.insert({key, value}).second)
            {
                values.emplace_back(key, value);
            }
        }
        if (parser.parseRBrace() || parser.parseGreater())
        {
            return {};
        }
        return DictAttr::getAlreadySorted(getContext(), values);
    }
    return {};
}

void pylir::Dialect::PylirDialect::printAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer) const
{
    llvm::TypeSwitch<mlir::Attribute>(attribute)
        .Case<NoneAttr>([&](NoneAttr) { printer << "none"; })
        .Case<NotImplementedAttr>([&](NotImplementedAttr) { printer << "notImplemented"; })
        .Case<FloatAttr>([&](FloatAttr attr) { printer << "float<" << attr.getValue() << ">"; })
        .Case<BoolAttr>([&](BoolAttr attr) { printer << "bool<" << (attr.getValue() ? "true" : "false") << ">"; })
        .Case<IntegerAttr>([&](IntegerAttr attr)
                           { printer << "integer<\"" << attr.getValue().toString(10, false) << "\">"; })
        .Case<StringAttr>([&](StringAttr attr) { printer << "string<\"" << attr.getValue() << "\">"; })
        .Case<ListAttr>(
            [&](ListAttr attr)
            {
                printer << "list<[";
                llvm::interleaveComma(attr.getValue(), printer);
                printer << "]>";
            })
        .Case<SetAttr>(
            [&](SetAttr attr)
            {
                printer << "set<{";
                llvm::interleaveComma(attr.getValue(), printer);
                printer << "}>";
            })
        .Case<DictAttr>(
            [&](DictAttr attr)
            {
                printer << "dict<{";
                llvm::interleaveComma(attr.getValue(), printer,
                                      [&](const auto& pair) { printer << pair.first << " : " << pair.second; });
                printer << "}>";
            });
}

mlir::Operation* pylir::Dialect::PylirDialect::materializeConstant(::mlir::OpBuilder& builder, ::mlir::Attribute value,
                                                                   ::mlir::Type type, ::mlir::Location loc)
{
    return builder.create<ConstantOp>(loc, type, value);
}

std::vector<mlir::Type> pylir::Dialect::detail::variantUnion(llvm::ArrayRef<mlir::Type> types)
{
    llvm::DenseSet<mlir::Type> unique;
    for (auto& iter : types)
    {
        if (auto variant = iter.dyn_cast_or_null<Dialect::VariantType>())
        {
            unique.insert(variant.getTypes().begin(), variant.getTypes().end());
        }
        else
        {
            unique.insert(iter);
        }
    }
    return {unique.begin(), unique.end()};
}

mlir::Type pylir::Dialect::VariantType::parse(::mlir::MLIRContext*, ::mlir::DialectAsmParser& parser)
{
    if (parser.parseLess())
    {
        return {};
    }
    llvm::SmallVector<mlir::Type> containedTypes;
    {
        Type containedType;
        if (parser.parseType(containedType))
        {
            return {};
        }
        containedTypes.push_back(std::move(containedType));
    }
    while (!parser.parseOptionalComma())
    {
        Type containedType;
        if (parser.parseType(containedType))
        {
            return {};
        }
        containedTypes.push_back(std::move(containedType));
    }
    if (parser.parseGreater())
    {
        return {};
    }
    return VariantType::get(containedTypes);
}

void pylir::Dialect::VariantType::print(::mlir::DialectAsmPrinter& printer) const
{
    printer << getMnemonic() << "<";
    llvm::interleaveComma(getTypes(), printer);
    printer << ">";
}

mlir::LogicalResult pylir::Dialect::VariantType::verifyConstructionInvariants(::mlir::Location loc,
                                                                              ::llvm::ArrayRef<::mlir::Type> types)
{
    if (types.empty())
    {
        return mlir::emitError(loc, "variant must contain at least one type");
    }
    llvm::DenseSet<mlir::Type> set(types.begin(), types.end());
    if (set.size() != types.size())
    {
        return mlir::emitError(loc, "variant contains duplicate type");
    }
    return mlir::success();
}

mlir::Type pylir::Dialect::SlotObjectType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser)
{
    if (parser.parseLBrace())
    {
        return {};
    }
    if (!parser.parseOptionalRBrace())
    {
        return SlotObjectType::get(context, {});
    }
    std::vector<std::pair<mlir::Attribute, mlir::Type>> vector;
    {
        mlir::Attribute key;
        mlir::Type value;
        if (parser.parseAttribute(key) || parser.parseColon() || parser.parseType(value))
        {
            return {};
        }
        vector.emplace_back(key, value);
    }
    while (!parser.parseComma())
    {
        mlir::Attribute key;
        mlir::Type value;
        if (parser.parseAttribute(key) || parser.parseColon() || parser.parseType(value))
        {
            return {};
        }
        vector.emplace_back(key, value);
    }
    if (parser.parseRBrace())
    {
        return {};
    }
    return SlotObjectType::get(context, vector);
}

void pylir::Dialect::SlotObjectType::print(::mlir::DialectAsmPrinter& printer) const
{
    printer << getMnemonic() << "{";
    llvm::interleaveComma(getSlots(), printer,
                          [&](const auto& pair) { printer << pair.first << " : " << pair.second; });
    printer << "}";
}

mlir::Type pylir::Dialect::KnownTypeObjectType::parse(::mlir::MLIRContext*, ::mlir::DialectAsmParser& parser)
{
    mlir::FlatSymbolRefAttr attribute;
    if (parser.parseLess() || parser.parseAttribute(attribute) || parser.parseGreater())
    {
        return {};
    }
    return KnownTypeObjectType::get(attribute);
}

void pylir::Dialect::KnownTypeObjectType::print(::mlir::DialectAsmPrinter& printer) const
{
    printer << getMnemonic() << '<' << getType() << '>';
}
