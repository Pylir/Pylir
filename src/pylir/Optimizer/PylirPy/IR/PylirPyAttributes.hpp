
#pragma once

#include <mlir/IR/Attributes.h>

#include <pylir/Support/BigInt.hpp>

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.h.inc"

namespace pylir::Py
{
class BoolAttr : public ::mlir::Attribute::AttrBase<BoolAttr, IntAttr, IntAttr::ImplType>
{
public:
    using Base::Base;

    static BoolAttr get(::mlir::MLIRContext* context, bool value);

    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return ::llvm::StringLiteral("bool");
    }

    static ::mlir::Attribute parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser, ::mlir::Type type);

    void print(::mlir::DialectAsmPrinter& printer) const;

    bool getValue() const;
};
} // namespace pylir::Py