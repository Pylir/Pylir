
#include "PylirPyDialect.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsDialect.cpp.inc"

#include "PylirPyAttributes.hpp"
#include "PylirPyOps.hpp"
#include "PylirPyTypes.hpp"

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
        auto count = mp_ubin_size(&key.getHandle());
        llvm::SmallVector<std::uint8_t, 10> data(count);
        auto result = mp_to_ubin(&key.getHandle(), data.data(), count, nullptr);
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

void pylir::Py::PylirPyDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"
        >();
    addAttributes<
#define GET_ATTRDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.cpp.inc"
        >();
}

mlir::Operation* pylir::Py::PylirPyDialect::materializeConstant(::mlir::OpBuilder& builder, ::mlir::Attribute value,
                                                                ::mlir::Type type, ::mlir::Location loc)
{
    return builder.create<Py::ConstantOp>(loc, type, value);
}

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.cpp.inc"

mlir::Type pylir::Py::PylirPyDialect::parseType(::mlir::DialectAsmParser& parser) const
{
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword))
    {
        return {};
    }
    mlir::Type result;
    (void)generatedTypeParser(getContext(), parser, keyword, result);
    return result;
}

void pylir::Py::PylirPyDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter& os) const
{
    (void)generatedTypePrinter(type, os);
}

mlir::Attribute pylir::Py::PylirPyDialect::parseAttribute(::mlir::DialectAsmParser& parser, ::mlir::Type type) const
{
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword))
    {
        return {};
    }
    mlir::Attribute result;
    (void)generatedAttributeParser(getContext(), parser, keyword, type, result);
    return result;
}

void pylir::Py::PylirPyDialect::printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter& os) const
{
    (void)generatedAttributePrinter(attr, os);
}

pylir::BigInt pylir::Py::IntAttr::getValue() const
{
    return getImpl()->value;
}

void pylir::Py::IntAttr::print(::mlir::DialectAsmPrinter& printer) const
{
    printer << getMnemonic() << "<" << getValue().toString() << ">";
}

mlir::Attribute pylir::Py::IntAttr::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser, ::mlir::Type)
{
    llvm::APInt apInt;
    if (parser.parseLess() || parser.parseInteger(apInt) || parser.parseGreater())
    {
        return {};
    }
    llvm::SmallString<10> str;
    apInt.toStringSigned(str);
    return IntAttr::get(context, BigInt({str.data(), str.size()}));
}

void pylir::Py::ListAttr::print(::mlir::DialectAsmPrinter& printer) const
{
    printer << getMnemonic() << "<[";
    llvm::interleaveComma(getValue(), printer);
    printer << "]>";
}

mlir::Attribute pylir::Py::ListAttr::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser, ::mlir::Type)
{
    if (parser.parseLess() || parser.parseLSquare())
    {
        return {};
    }
    if (!parser.parseOptionalRSquare())
    {
        if (parser.parseGreater())
        {
            return {};
        }
        return get(context, {});
    }
    llvm::SmallVector<mlir::Attribute> attrs;
    if (parser.parseAttribute(attrs.emplace_back()))
    {
        return {};
    }
    while (!parser.parseOptionalComma())
    {
        if (parser.parseAttribute(attrs.emplace_back()))
        {
            return {};
        }
    }
    if (parser.parseRSquare() || parser.parseGreater())
    {
        return {};
    }
    return get(context, attrs);
}

void pylir::Py::TupleAttr::print(::mlir::DialectAsmPrinter& printer) const
{
    printer << getMnemonic() << "<(";
    llvm::interleaveComma(getValue(), printer);
    printer << ")>";
}

mlir::Attribute pylir::Py::TupleAttr::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser,
                                            ::mlir::Type)
{
    if (parser.parseLess() || parser.parseLParen())
    {
        return {};
    }
    if (!parser.parseOptionalRParen())
    {
        if (parser.parseGreater())
        {
            return {};
        }
        return get(context, {});
    }
    llvm::SmallVector<mlir::Attribute> attrs;
    if (parser.parseAttribute(attrs.emplace_back()))
    {
        return {};
    }
    while (!parser.parseOptionalComma())
    {
        if (parser.parseAttribute(attrs.emplace_back()))
        {
            return {};
        }
    }
    if (parser.parseRParen() || parser.parseGreater())
    {
        return {};
    }
    return get(context, attrs);
}

void pylir::Py::SetAttr::print(::mlir::DialectAsmPrinter& printer) const
{
    printer << getMnemonic() << "<{";
    llvm::interleaveComma(getValue(), printer);
    printer << "}>";
}

mlir::Attribute pylir::Py::SetAttr::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser, ::mlir::Type)
{
    if (parser.parseLess() || parser.parseLBrace())
    {
        return {};
    }
    if (!parser.parseOptionalRBrace())
    {
        if (parser.parseGreater())
        {
            return {};
        }
        return get(context, {});
    }
    llvm::SmallVector<mlir::Attribute> attrs;
    if (parser.parseAttribute(attrs.emplace_back()))
    {
        return {};
    }
    while (!parser.parseOptionalComma())
    {
        if (parser.parseAttribute(attrs.emplace_back()))
        {
            return {};
        }
    }
    if (parser.parseRBrace() || parser.parseGreater())
    {
        return {};
    }
    return get(context, attrs);
}

void pylir::Py::DictAttr::print(::mlir::DialectAsmPrinter& printer) const
{
    printer << getMnemonic() << "<{";
    llvm::interleaveComma(getValue(), printer, [&](auto&& pair) { printer << pair.first << " to " << pair.second; });
    printer << "}>";
}

mlir::Attribute pylir::Py::DictAttr::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser, ::mlir::Type)
{
    if (parser.parseLess() || parser.parseLBrace())
    {
        return {};
    }
    if (!parser.parseOptionalRBrace())
    {
        if (parser.parseGreater())
        {
            return {};
        }
        return get(context, {});
    }
    llvm::SmallVector<std::pair<mlir::Attribute, mlir::Attribute>> attrs;
    if (parser.parseAttribute(attrs.emplace_back().first) || parser.parseKeyword("to")
        || parser.parseAttribute(attrs.back().second))
    {
        return {};
    }
    while (!parser.parseOptionalComma())
    {
        if (parser.parseAttribute(attrs.emplace_back().first) || parser.parseKeyword("to")
            || parser.parseAttribute(attrs.back().second))
        {
            return {};
        }
    }
    if (parser.parseRBrace() || parser.parseGreater())
    {
        return {};
    }
    return get(context, attrs);
}
