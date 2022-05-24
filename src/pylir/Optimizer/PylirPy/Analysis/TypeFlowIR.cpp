// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypeFlowIR.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/FunctionImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

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

mlir::OpFoldResult pylir::TypeFlow::UndefOp::fold(::llvm::ArrayRef<::mlir::Attribute>)
{
    return UndefAttr::get(getContext());
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

#define GET_OP_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIROps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRAttributes.cpp.inc"
