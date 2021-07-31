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
        >();
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
    if (type.getDialect().getTypeID() != getTypeID() && type.getDialect().getTypeID() == value.getDialect().getTypeID())
    {
        return type.getDialect().materializeConstant(builder, value, type, loc);
    }
    return builder.create<ConstantOp>(loc, type, value);
}

void pylir::Dialect::ObjectType::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                          llvm::function_ref<void(mlir::Type)>) const
{
    if (getType())
    {
        walkAttrsFn(getType());
    }
}

pylir::Dialect::DictAttr
    pylir::Dialect::DictAttr::getAlreadySorted(mlir::MLIRContext* context,
                                               llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value)
{
    return Base::get(context, value);
}

void pylir::Dialect::SetAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                       llvm::function_ref<void(mlir::Type)>) const
{
    std::for_each(getValue().begin(), getValue().end(), walkAttrsFn);
}

void pylir::Dialect::DictAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                        llvm::function_ref<void(mlir::Type)>) const
{
    std::for_each(getValue().begin(), getValue().end(),
                  [&](const auto& pair)
                  {
                      walkAttrsFn(pair.first);
                      walkAttrsFn(pair.second);
                  });
}

#include <pylir/Optimizer/Dialect/PylirOpsDialect.cpp.inc>
