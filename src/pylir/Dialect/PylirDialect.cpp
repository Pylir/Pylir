#include "PylirDialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Functional.hpp>

#include <unordered_set>

#include "PylirAttributes.hpp"
#include "PylirOps.hpp"
#include "PylirTypes.hpp"

#define GET_TYPEDEF_CLASSES
#include "pylir/Dialect/PylirOpsTypes.cpp.inc"

void pylir::Dialect::PylirDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "pylir/Dialect/PylirOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Dialect/PylirOpsTypes.cpp.inc"
        >();
    addAttributes<NoneAttr, BoolAttr, FloatAttr, IntegerAttr, StringAttr>();
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
    if (ref == "float")
    {
        double value;
        if (parser.parseFloat(value))
        {
            return {};
        }
        return FloatAttr::get(getContext(), value);
    }
    if (ref == "bool")
    {
        llvm::StringRef boolValue;
        if (parser.parseKeyword(&boolValue))
        {
            return {};
        }
        if (boolValue != "true" && boolValue != "false")
        {
            return {};
        }
        return BoolAttr::get(getContext(), boolValue == "true");
    }
    if (ref == "integer")
    {
        llvm::StringRef value;
        if (parser.parseOptionalString(&value))
        {
            return {};
        }
        llvm::APInt integer;
        if (value.getAsInteger(10, integer))
        {
            return {};
        }
        return IntegerAttr::get(getContext(), std::move(integer));
    }
    if (ref == "string")
    {
        llvm::StringRef value;
        if (parser.parseOptionalString(&value))
        {
            return {};
        }
        return StringAttr::get(getContext(), value.str());
    }
    return {};
}

void pylir::Dialect::PylirDialect::printAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer) const
{
    llvm::TypeSwitch<mlir::Attribute>(attribute)
        .Case<NoneAttr>([&](NoneAttr) { printer << "none"; })
        .Case<FloatAttr>([&](FloatAttr attr) { printer << "float " << attr.getValue(); })
        .Case<BoolAttr>([&](BoolAttr attr) { printer << "bool " << (attr.getValue() ? "true" : "false"); })
        .Case<IntegerAttr>([&](IntegerAttr attr)
                           { printer << "integer \"" << attr.getValue().toString(10, false) << "\""; })
        .Case<StringAttr>([&](StringAttr attr) { printer << "string \"" << attr.getValue() << "\""; });
}

mlir::Operation* pylir::Dialect::PylirDialect::materializeConstant(::mlir::OpBuilder& builder, ::mlir::Attribute value,
                                                                   ::mlir::Type type, ::mlir::Location loc)
{
    return builder.create<ConstantOp>(loc, type, value);
}

mlir::Type pylir::Dialect::VariantType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser)
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
    return VariantType::get(context, containedTypes);
}

void pylir::Dialect::VariantType::print(::mlir::DialectAsmPrinter& printer) const
{
    printer << "variant<";
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
    struct MLIRHash
    {
        std::size_t operator()(mlir::Type type) const noexcept
        {
            return mlir::hash_value(type);
        }
    };
    std::unordered_set<mlir::Type, MLIRHash> set(types.begin(), types.end());
    if (set.size() != types.size())
    {
        return mlir::emitError(loc, "variant contains duplicate type");
    }
    return mlir::success();
}
