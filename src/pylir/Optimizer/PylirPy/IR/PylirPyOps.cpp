#include "PylirPyOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Text.hpp>

#include "PylirPyAttributes.hpp"

#define GET_OP_CLASSES
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc>

mlir::OpFoldResult pylir::Py::ConstantOp::fold(::llvm::ArrayRef<::mlir::Attribute>)
{
    return constant();
}

mlir::OpFoldResult pylir::Py::GetAttrOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 1);
    if (!operands[0])
    {
        return nullptr;
    }
    return llvm::TypeSwitch<mlir::Attribute, mlir::OpFoldResult>(operands[0]).Default(mlir::OpFoldResult{nullptr});
}

mlir::OpFoldResult pylir::Py::GetItemOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    PYLIR_ASSERT(operands.size() == 2);
    auto object = operands[0];
    auto index = operands[1];
    if (!object || !index)
    {
        return nullptr;
    }
    return llvm::TypeSwitch<mlir::Attribute, mlir::OpFoldResult>(operands[0])
        .Case(
            [&](mlir::StringAttr attr) -> mlir::OpFoldResult
            {
                auto integer = index.dyn_cast_or_null<mlir::IntegerAttr>();
                if (!integer)
                {
                    return nullptr;
                }
                auto utf32 = Text::toUTF32String(attr.getValue());
                auto value = integer.getValue();
                if (value.isNegative())
                {
                    value += utf32.size();
                }
                if (value.isNegative() || value.sge(utf32.size()))
                {
                    return nullptr;
                }
                auto codepoint = utf32[value.getZExtValue()];
                auto utf8 = Text::toUTF8String({&codepoint, 1});
                return mlir::StringAttr::get(getContext(), utf8);
            })
        .Case<Py::TupleAttr, Py::ListAttr>(
            [&](auto sequence) -> mlir::OpFoldResult
            {
                auto array = sequence.getValue();
                auto integer = index.dyn_cast_or_null<mlir::IntegerAttr>();
                if (!integer)
                {
                    return nullptr;
                }
                auto value = integer.getValue();
                if (value.isNegative())
                {
                    value += array.size();
                }
                if (value.isNegative() || value.sge(array.size()))
                {
                    return nullptr;
                }
                return array[value.getZExtValue()];
            })
        .Case(
            [&](Py::DictAttr dictAttr) -> mlir::OpFoldResult
            {
                auto result = std::find_if(dictAttr.getValue().begin(), dictAttr.getValue().end(),
                                           [&](auto&& pair) { return pair.first == index; });
                if (result != dictAttr.getValue().end())
                {
                    return result->second;
                }
                return nullptr;
            })
        .Default(mlir::OpFoldResult{nullptr});
}

mlir::CallInterfaceCallable pylir::Py::CallOp::getCallableForCallee()
{
    return callee();
}

mlir::Operation::operand_range pylir::Py::CallOp::getArgOperands()
{
    return arguments();
}
