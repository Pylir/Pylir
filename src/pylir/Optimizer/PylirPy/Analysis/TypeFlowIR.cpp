// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypeFlowIR.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/FunctionImplementation.h>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Optimizer/PylirPy/IR/ObjectAttrInterface.hpp>
#include <pylir/Optimizer/PylirPy/Interfaces/TypeRefineableInterface.hpp>

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

mlir::LogicalResult pylir::TypeFlow::TypeOfOp::exec(::llvm::ArrayRef<::mlir::Attribute> operands,
                                                    ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results,
                                                    ::mlir::SymbolTableCollection& collection)
{
    if (auto typeAttr = operands[0].dyn_cast_or_null<mlir::TypeAttr>())
    {
        results.emplace_back(typeAttr.getValue().cast<Py::ObjectTypeInterface>().getTypeObject());
        return mlir::success();
    }
    if (operands[0].isa_and_nonnull<Py::ObjectAttrInterface, mlir::SymbolRefAttr>())
    {
        results.emplace_back(Py::typeOfConstant(operands[0], collection, getContext()).getTypeObject());
        return mlir::success();
    }
    if (auto makeObject = getInput().getDefiningOp<MakeObjectOp>())
    {
        results.emplace_back(makeObject.getInput());
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::TypeFlow::MakeObjectOp::exec(::llvm::ArrayRef<::mlir::Attribute> operands,
                                                        ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results,
                                                        ::mlir::SymbolTableCollection&)
{
    if (auto ref = operands[0].dyn_cast_or_null<mlir::FlatSymbolRefAttr>())
    {
        results.emplace_back(mlir::TypeAttr::get(Py::ClassType::get(ref)));
        return mlir::success();
    }
    if (auto typeOf = getInput().getDefiningOp<TypeOfOp>())
    {
        results.emplace_back(typeOf.getInput());
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult pylir::TypeFlow::CalcOp::exec(::llvm::ArrayRef<::mlir::Attribute> operands,
                                                  ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results,
                                                  ::mlir::SymbolTableCollection& collection)
{
    if (mlir::succeeded(getInstruction()->fold(operands, results)) && !results.empty())
    {
        if (!getValueCalc())
        {
            for (auto& iter : results)
            {
                if (auto attr = iter.dyn_cast<mlir::Attribute>())
                {
                    iter = mlir::TypeAttr::get(pylir::Py::typeOfConstant(attr, collection, getInstruction()));
                }
            }
        }
        return mlir::success();
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
    llvm::SmallVector<pylir::Py::ObjectTypeInterface> inputTypes(operands.size());
    for (auto [type, operand] : llvm::zip(inputTypes, operands))
    {
        if (auto typeAttr = operand.dyn_cast_or_null<mlir::TypeAttr>())
        {
            type = typeAttr.getValue().dyn_cast<pylir::Py::ObjectTypeInterface>();
        }
        else if (operand.isa_and_nonnull<pylir::Py::ObjectAttrInterface, mlir::SymbolRefAttr>())
        {
            type = pylir::Py::typeOfConstant(operand, collection, getInstruction());
        }
    }
    llvm::SmallVector<pylir::Py::ObjectTypeInterface> resultTypes;
    if (refinable.refineTypes(inputTypes, resultTypes, collection) != Py::TypeRefineResult::Success)
    {
        return mlir::failure();
    }
    results.resize(resultTypes.size());
    for (auto [foldRes, resType] : llvm::zip(results, resultTypes))
    {
        foldRes = mlir::TypeAttr::get(resType);
    }
    return mlir::success();
}

mlir::LogicalResult pylir::TypeFlow::CalcOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands,
                                                  ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results)
{
    mlir::SymbolTableCollection collection;
    return exec(operands, results, collection);
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
    if (mlir::isa<pylir::Py::TypeRefineableInterface>(adaptor.getInstruction()))
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

mlir::Value pylir::TypeFlow::TypeOfOp::mapValue(::mlir::Value resultValue)
{
    return getContext()->getResult(resultValue.cast<mlir::OpResult>().getResultNumber());
}

mlir::Value pylir::TypeFlow::CalcOp::mapValue(::mlir::Value resultValue)
{
    return getInstruction()->getResult(resultValue.cast<mlir::OpResult>().getResultNumber());
}

mlir::Value pylir::TypeFlow::CallOp::mapValue(::mlir::Value resultValue)
{
    return getContext()->getResult(resultValue.cast<mlir::OpResult>().getResultNumber());
}

mlir::Value pylir::TypeFlow::CallIndirectOp::mapValue(::mlir::Value resultValue)
{
    return getContext()->getResult(resultValue.cast<mlir::OpResult>().getResultNumber());
}

#define GET_OP_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIROps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRAttributes.cpp.inc"
