#include "PylirPyOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Text.hpp>

#include "PylirPyAttributes.hpp"

namespace
{
bool parseExpandArguments(mlir::OpAsmParser& parser, llvm::SmallVectorImpl<mlir::OpAsmParser::OperandType>& operands,
                          mlir::ArrayAttr& iterExpansion, mlir::ArrayAttr& mappingExpansion)
{
    llvm::SmallVector<std::int32_t> iters;
    llvm::SmallVector<std::int32_t> mappings;
    auto exit = llvm::make_scope_exit(
        [&]
        {
            iterExpansion = parser.getBuilder().getI32ArrayAttr(iters);
            mappingExpansion = parser.getBuilder().getI32ArrayAttr(mappings);
        });

    if (parser.parseLParen())
    {
        return true;
    }
    if (!parser.parseOptionalRParen())
    {
        return false;
    }

    std::int32_t index = 0;
    auto parseOnce = [&]
    {
        if (!parser.parseOptionalStar())
        {
            if (!parser.parseOptionalStar())
            {
                mappings.push_back(index++);
            }
            else
            {
                iters.push_back(index++);
            }
        }
        return parser.parseOperand(operands.emplace_back());
    };
    if (parseOnce())
    {
        return true;
    }
    while (!parser.parseOptionalComma())
    {
        if (parseOnce())
        {
            return true;
        }
    }

    if (parser.parseRParen())
    {
        return true;
    }

    return false;
}

void printExpandArguments(mlir::OpAsmPrinter& printer, mlir::Operation*, mlir::OperandRange operands,
                          mlir::ArrayAttr iterExpansion, mlir::ArrayAttr mappingExpansion)
{
    printer << '(';
    llvm::DenseSet<std::uint32_t> iters;
    for (auto iter : iterExpansion.getAsValueRange<mlir::IntegerAttr>())
    {
        iters.insert(iter.getZExtValue());
    }
    llvm::DenseSet<std::uint32_t> mappings;
    for (auto iter : mappingExpansion.getAsValueRange<mlir::IntegerAttr>())
    {
        mappings.insert(iter.getZExtValue());
    }
    int i = 0;
    llvm::interleaveComma(operands, printer,
                          [&](mlir::Value value)
                          {
                              if (iters.contains(i))
                              {
                                  printer << '*' << value;
                              }
                              else if (mappings.contains(i))
                              {
                                  printer << "**" << value;
                              }
                              else
                              {
                                  printer << value;
                              }
                              i++;
                          });
    printer << ')';
}

} // namespace

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
