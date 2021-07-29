#include "PylirDialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Functional.hpp>

#include "PylirOps.hpp"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/Dialect/PylirOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/Dialect/PylirOpsAttributes.cpp.inc"

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
        return mlir::success();
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
    addAttributes<
#define GET_ATTRDEF_LIST
#include "pylir/Optimizer/Dialect/PylirOpsAttributes.cpp.inc"
        , SetAttr, DictAttr>();
}

mlir::Type pylir::Dialect::PylirDialect::parseType(::mlir::DialectAsmParser& parser) const
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

void pylir::Dialect::PylirDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter& os) const
{
    auto result = generatedTypePrinter(type, os);
    PYLIR_ASSERT(mlir::succeeded(result));
}

mlir::Attribute pylir::Dialect::PylirDialect::parseAttribute(::mlir::DialectAsmParser& parser, ::mlir::Type type) const
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

void pylir::Dialect::PylirDialect::printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter& os) const
{
    auto result = generatedAttributePrinter(attr, os);
    PYLIR_ASSERT(mlir::succeeded(result));
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

pylir::Dialect::DictAttr
    pylir::Dialect::DictAttr::getAlreadySorted(mlir::MLIRContext* context,
                                               llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value)
{
    return Base::get(context, value);
}
