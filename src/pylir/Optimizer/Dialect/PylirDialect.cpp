#include "PylirDialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Functional.hpp>

#include "PylirOps.hpp"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/Dialect/PylirOpsTypes.cpp.inc"

namespace pylir::Dialect::detail
{
struct ObjectTypeStorage : public mlir::TypeStorage
{
    using KeyTy = std::tuple<>;

    mlir::FlatSymbolRefAttr type;

    bool operator==(const KeyTy&) const
    {
        return true;
    }

    static ObjectTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy&)
    {
        return new (allocator.allocate<ObjectTypeStorage>()) ObjectTypeStorage();
    }

    mlir::LogicalResult mutate(mlir::TypeStorageAllocator&, mlir::FlatSymbolRefAttr newType)
    {
        type = newType;
        return success();
    }
};
} // namespace pylir::Dialect::detail

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
    addAttributes<IntegerAttr, StringAttr, SetAttr, DictAttr>();
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
        .Case<IntegerAttr>([&](IntegerAttr attr)
                           { printer << "integer<\"" << attr.getValue().toString(10, false) << "\">"; })
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

void pylir::Dialect::ObjectType::setKnownType(mlir::FlatSymbolRefAttr type)
{
    PYLIR_ASSERT(type);
    auto result = Base::mutate(type);
    PYLIR_ASSERT(mlir::succeeded(result));
}

void pylir::Dialect::ObjectType::clearType()
{
    auto result = Base::mutate(mlir::FlatSymbolRefAttr{});
    PYLIR_ASSERT(mlir::succeeded(result));
}
