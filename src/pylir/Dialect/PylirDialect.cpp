#include "PylirDialect.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

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
#include "pylir/Dialect/PylirOpsTypes.h.inc"
        >();
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

mlir::Type pylir::Dialect::PylirVariantType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser)
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
    return PylirVariantType::get(context, containedTypes);
}

void pylir::Dialect::PylirVariantType::print(::mlir::DialectAsmPrinter& printer) const
{
    printer << "tuple<";
    llvm::interleaveComma(getTypes(), printer);
    printer << ">";
}

mlir::LogicalResult pylir::Dialect::PylirVariantType::verifyConstructionInvariants(::mlir::Location loc,
                                                                                   ::llvm::ArrayRef<::mlir::Type> types)
{
    if (types.empty())
    {
        return mlir::emitError(loc, "variant must contain at least one type");
    }
    return mlir::success();
}
