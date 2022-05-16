// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypeFlowIR.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/FunctionImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

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

mlir::SuccessorOperands pylir::TypeFlow::BranchOp::getSuccessorOperands(unsigned int index)
{
    return mlir::SuccessorOperands(getBranchArgsMutable()[index]);
}

mlir::LogicalResult
    pylir::TypeFlow::RefineableOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location>,
                                                    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
                                                    ::mlir::RegionRange regions,
                                                    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    Adaptor adaptor(operands, attributes, regions);
    inferredReturnTypes.resize(adaptor.getInstruction()->getNumResults(), TypeFlow::MetaType::get(context));
    return mlir::success();
}

mlir::LogicalResult
    pylir::TypeFlow::FoldableOp::inferReturnTypes(::mlir::MLIRContext*, ::llvm::Optional<::mlir::Location>,
                                                  ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
                                                  ::mlir::RegionRange regions,
                                                  ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes)
{
    Adaptor adaptor(operands, attributes, regions);
    inferredReturnTypes.append(llvm::to_vector(adaptor.getInstruction()->getResultTypes()));
    return mlir::success();
}

#define GET_OP_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIROps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/Analysis/TypeFlowIRAttributes.cpp.inc"
