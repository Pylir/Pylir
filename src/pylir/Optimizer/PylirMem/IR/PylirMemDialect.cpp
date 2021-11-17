#include "PylirMemDialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Functional.hpp>

#include "PylirMemOps.hpp"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirMem/IR/PylirMemOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirMem/IR/PylirMemOpsAttributes.cpp.inc"

void pylir::Mem::PylirMemDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "pylir/Optimizer/PylirMem/IR/PylirMemOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Optimizer/PylirMem/IR/PylirMemOpsTypes.cpp.inc"
        >();
    addAttributes<
#define GET_ATTRDEF_LIST
#include "pylir/Optimizer/PylirMem/IR/PylirMemOpsAttributes.cpp.inc"
        >();
}

/*
mlir::Type pylir::Mem::PylirMemDialect::parseType(::mlir::DialectAsmParser& parser) const
{
    llvm::StringRef ref;
    if (parser.parseKeyword(&ref))
    {
        return {};
    }
    mlir::Type type;
    generatedTypeParser(getContext(), parser, ref, type);
    return type;
}

void pylir::Mem::PylirMemDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter& os) const
{
    auto result = generatedTypePrinter(type, os);
    PYLIR_ASSERT(mlir::succeeded(result));
}

mlir::Attribute pylir::Mem::PylirMemDialect::parseAttribute(::mlir::DialectAsmParser& parser, ::mlir::Type type) const
{
    llvm::StringRef ref;
    if (parser.parseKeyword(&ref))
    {
        return {};
    }
    mlir::Attribute attribute;
    generatedAttributeParser(getContext(), parser, ref, type, attribute);
    return attribute;
}

void pylir::Mem::PylirMemDialect::printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter& os) const
{
    auto result = generatedAttributePrinter(attr, os);
    PYLIR_ASSERT(mlir::succeeded(result));
}
*/

#include <pylir/Optimizer/PylirMem/IR/PylirMemOpsDialect.cpp.inc>
