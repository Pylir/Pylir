//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypeFlowIR.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/FunctionImplementation.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/IR/ObjectAttrInterface.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyAttributes.hpp>
#include <pylir/Optimizer/PylirPy/IR/TypeRefineableInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/ObjectFromTypeObjectInterface.hpp>

void pylir::TypeFlow::TypeFlowDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIROps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRTypes.cpp.inc"
        >();
    addAttributes<
#define GET_ATTRDEF_LIST
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRAttributes.cpp.inc"
        >();
}

#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRDialect.cpp.inc"

void pylir::TypeFlow::InstructionAttr::print(::mlir::AsmPrinter& printer) const
{
    printer.getStream() << "// ";
    if (getInstruction()->getNumRegions() != 0)
    {
        printer.getStream() << getInstruction()->getRegisteredInfo()->getStringRef();
        return;
    }
    getInstruction()->print(printer.getStream(), mlir::OpPrintingFlags{}.useLocalScope());
}

mlir::Attribute pylir::TypeFlow::InstructionAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    parser.emitError(parser.getCurrentLocation(), "Parsing " + getMnemonic() + " not supported");
    return {};
}

mlir::ParseResult pylir::TypeFlow::FuncOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result)
{
    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, false,
        [](mlir::Builder& builder, llvm::ArrayRef<mlir::Type> argTypes, llvm::ArrayRef<mlir::Type> results, auto&&...)
        { return builder.getFunctionType(argTypes, results); });
}

void pylir::TypeFlow::FuncOp::print(::mlir::OpAsmPrinter& p)
{
    mlir::function_interface_impl::printFunctionOp(p, *this, false);
}

mlir::OpFoldResult pylir::TypeFlow::ConstantOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    return operands[0];
}

mlir::LogicalResult pylir::TypeFlow::TypeOfOp::exec(::llvm::ArrayRef<Py::TypeAttrUnion> operands,
                                                    ::llvm::SmallVectorImpl<OpFoldResult>& results,
                                                    ::mlir::SymbolTableCollection& collection)
{
    if (auto type = operands[0].dyn_cast_or_null<Py::ObjectTypeInterface>())
    {
        results.emplace_back(type.getTypeObject());
        return mlir::success();
    }
    if (operands[0].isa_and_nonnull<Py::ObjectAttrInterface, Py::RefAttr, Py::UnboundAttr>())
    {
        results.emplace_back(
            Py::typeOfConstant(operands[0].cast<mlir::Attribute>(), collection, getInstruction()).getTypeObject());
        return mlir::success();
    }
    return mlir::failure();
}

namespace
{
mlir::Value mapBackOperand(mlir::Value foldResult, mlir::Operation* normalIROp, mlir::Operation* typeFlowIROp)
{
    // A value returned by the fold operation has to somewhat be mapped back to a value within the TypeFlowIR.
    // This is not 100% possible all the time, but we do so on best effort basis. If it is not possible
    // we can't use the fold result.
    auto* res = llvm::find_if(normalIROp->getOpOperands(),
                              [foldResult](mlir::OpOperand& operand) { return operand.get() == foldResult; });
    if (res == normalIROp->getOpOperands().end())
    {
        return {};
    }
    return typeFlowIROp->getOperand(res->getOperandNumber());
}
} // namespace

mlir::LogicalResult pylir::TypeFlow::TupleLenOp::exec(::llvm::ArrayRef<::pylir::Py::TypeAttrUnion> operands,
                                                      ::llvm::SmallVectorImpl<::pylir::TypeFlow::OpFoldResult>& results,
                                                      ::mlir::SymbolTableCollection&)
{
    llvm::SmallVector<mlir::OpFoldResult> mlirFoldResults;
    if (mlir::succeeded(getInstruction()->fold(
            llvm::to_vector(llvm::map_range(operands, [](Py::TypeAttrUnion value)
                                            { return value.dyn_cast_or_null<mlir::Attribute>(); })),
            mlirFoldResults))
        && !mlirFoldResults.empty())
    {
        bool success = true;
        for (auto [iter, type] : llvm::zip(mlirFoldResults, getInstruction()->getResultTypes()))
        {
            if (auto attr = iter.dyn_cast<mlir::Attribute>())
            {
                results.emplace_back(attr);
                continue;
            }
            auto value = mapBackOperand(iter.get<mlir::Value>(), getInstruction(), *this);
            if (!value)
            {
                success = false;
                break;
            }
            results.emplace_back(value);
        }
        if (success)
        {
            return mlir::success();
        }
    }
    if (auto type = operands[0].dyn_cast_or_null<pylir::Py::TupleType>())
    {
        results.emplace_back(mlir::IntegerAttr::get(getInstruction()->getResultTypes()[0], type.getElements().size()));
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::TypeFlow::IsOp::exec(::llvm::ArrayRef<::pylir::Py::TypeAttrUnion> operands,
                                                ::llvm::SmallVectorImpl<::pylir::TypeFlow::OpFoldResult>& results,
                                                ::mlir::SymbolTableCollection& collection)
{
    if (auto lhsAttr = operands[0].dyn_cast_or_null<mlir::Attribute>())
    {
        if (auto rhsAttr = operands[1].dyn_cast_or_null<mlir::Attribute>())
        {
            if (lhsAttr != rhsAttr)
            {
                results.emplace_back(mlir::BoolAttr::get(getContext(), false));
                return mlir::success();
            }
            if (lhsAttr.isa<Py::RefAttr>() && rhsAttr.isa<Py::RefAttr>())
            {
                results.emplace_back(mlir::BoolAttr::get(getContext(), true));
                return mlir::success();
            }
            return mlir::failure();
        }
    }
    auto lhsType = operands[0].dyn_cast_or_null<pylir::Py::ObjectTypeInterface>();
    auto rhsType = operands[1].dyn_cast_or_null<pylir::Py::ObjectTypeInterface>();
    if (!lhsType && operands[0].isa_and_nonnull<Py::UnboundAttr, Py::ObjectAttrInterface, Py::RefAttr>())
    {
        lhsType = Py::typeOfConstant(operands[0].cast<mlir::Attribute>(), collection, getInstruction());
    }
    if (!rhsType && operands[1].isa_and_nonnull<Py::UnboundAttr, Py::ObjectAttrInterface, Py::RefAttr>())
    {
        rhsType = Py::typeOfConstant(operands[1].cast<mlir::Attribute>(), collection, getInstruction());
    }
    if (!lhsType || !rhsType)
    {
        return mlir::failure();
    }
    if (rhsType != lhsType)
    {
        results.emplace_back(mlir::BoolAttr::get(getContext(), false));
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::TypeFlow::CalcOp::exec(::llvm::ArrayRef<Py::TypeAttrUnion> operands,
                                                  ::llvm::SmallVectorImpl<OpFoldResult>& results,
                                                  ::mlir::SymbolTableCollection& collection, bool forFolding)
{
    llvm::SmallVector<mlir::OpFoldResult> mlirFoldResults;
    if (mlir::succeeded(getInstruction()->fold(
            llvm::to_vector(llvm::map_range(operands, [](Py::TypeAttrUnion value)
                                            { return value.dyn_cast_or_null<mlir::Attribute>(); })),
            mlirFoldResults))
        && !mlirFoldResults.empty())
    {
        bool success = true;
        for (auto [iter, type] : llvm::zip(mlirFoldResults, getInstruction()->getResultTypes()))
        {
            // If this is a calc then all types that aren't `!py.dynamic` are ignored.
            if (!getValueCalc() && !type.isa<Py::DynamicType>())
            {
                continue;
            }
            if (auto attr = iter.dyn_cast<mlir::Attribute>())
            {
                if (!getValueCalc())
                {
                    results.emplace_back(pylir::Py::typeOfConstant(attr, collection, getInstruction()));
                    continue;
                }
                results.emplace_back(attr);
                continue;
            }
            auto value = mapBackOperand(iter.get<mlir::Value>(), getInstruction(), *this);
            if (!value)
            {
                success = false;
                break;
            }
            results.emplace_back(value);
        }
        if (success)
        {
            return mlir::success();
        }
    }
    if (getValueCalc())
    {
        return mlir::failure();
    }

    auto refinable = mlir::dyn_cast<pylir::Py::TypeRefineableInterface>(getInstruction());
    if (!refinable)
    {
        return mlir::failure();
    }
    auto inputs = llvm::to_vector(operands);
    auto objectFromTypeObject = mlir::dyn_cast<pylir::Py::ObjectFromTypeObjectInterface>(getInstruction());
    std::optional<std::size_t> exemptIndex =
        objectFromTypeObject ? std::optional{objectFromTypeObject.getTypeObjectIndex()} : std::nullopt;
    for (auto [iter, type] : llvm::zip(inputs, llvm::enumerate(getInstruction()->getOperandTypes())))
    {
        if (!type.value().isa<pylir::Py::DynamicType>() || type.index() == exemptIndex)
        {
            continue;
        }
        if (iter.isa_and_nonnull<pylir::Py::ObjectAttrInterface, Py::RefAttr, Py::UnboundAttr>())
        {
            iter = pylir::Py::typeOfConstant(iter.cast<mlir::Attribute>(), collection, getInstruction());
        }
    }
    llvm::SmallVector<pylir::Py::ObjectTypeInterface> resultTypes;
    auto result = refinable.refineTypes(inputs, resultTypes, collection);
    if (forFolding && result != Py::TypeRefineResult::Success)
    {
        return mlir::failure();
    }
    if (!forFolding && result == Py::TypeRefineResult::Failure)
    {
        return mlir::failure();
    }
    results.resize(resultTypes.size());
    std::copy(resultTypes.begin(), resultTypes.end(), results.begin());
    return mlir::success();
}

mlir::LogicalResult pylir::TypeFlow::CalcOp::exec(::llvm::ArrayRef<Py::TypeAttrUnion> operands,
                                                  ::llvm::SmallVectorImpl<OpFoldResult>& results,
                                                  ::mlir::SymbolTableCollection& collection)
{
    return exec(operands, results, collection, false);
}

mlir::LogicalResult pylir::TypeFlow::CalcOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands,
                                                  ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    llvm::SmallVector<::pylir::TypeFlow::OpFoldResult> res;
    mlir::SymbolTableCollection collection;
    if (mlir::failed(exec(
            llvm::to_vector(llvm::map_range(operands,
                                            [](mlir::Attribute attr) -> Py::TypeAttrUnion
                                            {
                                                if (auto typeAttr = attr.dyn_cast_or_null<mlir::TypeAttr>())
                                                {
                                                    return typeAttr.getValue().cast<pylir::Py::ObjectTypeInterface>();
                                                }
                                                return attr;
                                            })),
            res, collection, true)))
    {
        return mlir::failure();
    }
    for (auto& iter : res)
    {
        if (auto typeFlowValue = iter.dyn_cast<::pylir::Py::TypeAttrUnion>())
        {
            if (auto type = typeFlowValue.dyn_cast_or_null<pylir::Py::ObjectTypeInterface>())
            {
                results.emplace_back(mlir::TypeAttr::get(type));
                continue;
            }
            results.emplace_back(typeFlowValue.dyn_cast<mlir::Attribute>());
            continue;
        }
        results.emplace_back(iter.get<mlir::Value>());
    }
    return mlir::success();
}

mlir::SuccessorOperands pylir::TypeFlow::BranchOp::getSuccessorOperands(unsigned int index)
{
    return mlir::SuccessorOperands(getBranchArgsMutable()[index]);
}

mlir::SuccessorOperands pylir::TypeFlow::CondBranchOp::getSuccessorOperands(unsigned int index)
{
    return mlir::SuccessorOperands(index == 0 ? getTrueArgsMutable() : getFalseArgsMutable());
}

mlir::LogicalResult
    pylir::TypeFlow::CalcOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>,
                                              ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
                                              ::mlir::RegionRange regions,
                                              ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    Adaptor adaptor(operands, attributes, regions);
    std::size_t count = adaptor.getInstruction()->getNumResults();
    if (!adaptor.getValueCalc())
    {
        count = llvm::count_if(adaptor.getInstruction()->getResultTypes(),
                               std::mem_fn(&mlir::Type::isa<pylir::Py::DynamicType>));
    }
    inferredReturnTypes.resize(count, TypeFlow::MetaType::get(context));
    return mlir::success();
}

mlir::CallInterfaceCallable pylir::TypeFlow::CallOp::getCallableForCallee()
{
    return getCalleeAttr();
}

mlir::CallInterfaceCallable pylir::TypeFlow::CallIndirectOp::getCallableForCallee()
{
    return getCallee();
}

mlir::Operation::operand_range pylir::TypeFlow::CallOp::getArgOperands()
{
    return getArguments();
}

mlir::Operation::operand_range pylir::TypeFlow::CallIndirectOp::getArgOperands()
{
    return getArguments();
}

#define GET_OP_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIROps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRAttributes.cpp.inc"
