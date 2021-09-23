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

void pylir::Mem::PylirDialect::initialize()
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

mlir::Type pylir::Mem::PylirDialect::parseType(::mlir::DialectAsmParser& parser) const
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

void pylir::Mem::PylirDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter& os) const
{
    auto result = generatedTypePrinter(type, os);
    PYLIR_ASSERT(mlir::succeeded(result));
}

mlir::Attribute pylir::Mem::PylirDialect::parseAttribute(::mlir::DialectAsmParser& parser, ::mlir::Type type) const
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

void pylir::Mem::PylirDialect::printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter& os) const
{
    auto result = generatedAttributePrinter(attr, os);
    PYLIR_ASSERT(mlir::succeeded(result));
}

void pylir::Mem::PointerType::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)>,
                                                       llvm::function_ref<void(mlir::Type)> walkTypeFn) const
{
    walkTypeFn(getElementType());
}

pylir::Mem::DictAttr
    pylir::Mem::DictAttr::getAlreadySorted(mlir::MLIRContext* context,
                                           llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> value)
{
    return Base::get(context, value);
}

void pylir::Mem::SetAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                   llvm::function_ref<void(mlir::Type)>) const
{
    std::for_each(getValue().begin(), getValue().end(), walkAttrsFn);
}

void pylir::Mem::DictAttr::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                    llvm::function_ref<void(mlir::Type)>) const
{
    std::for_each(getValue().begin(), getValue().end(),
                  [&](const auto& pair)
                  {
                      walkAttrsFn(pair.first);
                      walkAttrsFn(pair.second);
                  });
}

#include <pylir/Optimizer/PylirMem/IR/PylirMemOpsDialect.cpp.inc>