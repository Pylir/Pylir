
#include "PylirTypes.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#define GET_TYPEDEF_CLASSES
#include "pylir/Dialect/PylirOpsTypes.cpp.inc"

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
