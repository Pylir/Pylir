#include "PylirPyTypes.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include "PylirPyDialect.hpp"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"

mlir::Type pylir::Py::PylirPyDialect::parseType(::mlir::DialectAsmParser& parser) const
{
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword))
    {
        return {};
    }
    mlir::Type result;
    (void)generatedTypeParser(parser, keyword, result);
    return result;
}

void pylir::Py::PylirPyDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter& os) const
{
    (void)generatedTypePrinter(type, os);
}
