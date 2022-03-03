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
    return getBranchArgsMutable()[index];
}

mlir::LogicalResult pylir::MemSSA::MemoryBranchOp::verify()
{
    if (getBranchArgs().size() != getSuccessors().size())
    {
        return emitOpError("Expected branch arguments for every successor");
    }
    return mlir::success();
}

void pylir::MemSSA::InstructionAttr::print(::mlir::AsmPrinter& printer) const
{
    printer.getStream() << "// ";
    if (getInstruction()->getNumRegions() != 0)
    {
        printer.getStream() << getInstruction()->getRegisteredInfo()->getStringRef();
        return;
    }
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

mlir::RegionKind pylir::MemSSA::MemoryModuleOp::getRegionKind(unsigned int)
{
    return mlir::RegionKind::SSACFG;
}

llvm::StringRef pylir::MemSSA::MemoryModuleOp::getDefaultDialect()
{
    return pylir::MemSSA::MemorySSADialect::getDialectNamespace();
}

#define GET_OP_CLASSES
#include "pylir/Optimizer/Analysis/MemorySSAIROps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/Analysis/MemorySSAIRTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/Analysis/MemorySSAIRAttributes.cpp.inc"
