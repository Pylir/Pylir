#include "MemorySSAIR.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

void pylir::MemSSA::MemorySSADialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "pylir/Optimizer/Analysis/MemorySSAIROps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Optimizer/Analysis/MemorySSAIRTypes.cpp.inc"
        >();
    addAttributes<
#define GET_ATTRDEF_LIST
#include "pylir/Optimizer/Analysis/MemorySSAIRAttributes.cpp.inc"
        >();
}

#include "pylir/Optimizer/Analysis/MemorySSAIRDialect.cpp.inc"

mlir::Optional<mlir::MutableOperandRange> pylir::MemSSA::MemoryBranchOp::getMutableSuccessorOperands(unsigned index)
{
    return branchArgsMutable()[index];
}

void pylir::MemSSA::InstructionAttr::print(::mlir::AsmPrinter& printer) const
{
    printer.getStream() << "// ";
    getInstruction()->print(printer.getStream(), mlir::OpPrintingFlags{}.useLocalScope());
}

mlir::Attribute pylir::MemSSA::InstructionAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    parser.emitError(parser.getCurrentLocation(), "Parsing " + getMnemonic() + " not supported");
    return {};
}

void pylir::MemSSA::ReadAttr::print(::mlir::AsmPrinter&) const {}

mlir::Attribute pylir::MemSSA::ReadAttr::parse(::mlir::AsmParser& parser, ::mlir::Type)
{
    parser.emitError(parser.getCurrentLocation(), "Parsing " + getMnemonic() + " not supported");
    return {};
}

mlir::RegionKind pylir::MemSSA::MemoryRegionOp::getRegionKind(unsigned int)
{
    return mlir::RegionKind::SSACFG;
}

#define GET_OP_CLASSES
#include "pylir/Optimizer/Analysis/MemorySSAIROps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/Analysis/MemorySSAIRTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/Analysis/MemorySSAIRAttributes.cpp.inc"
