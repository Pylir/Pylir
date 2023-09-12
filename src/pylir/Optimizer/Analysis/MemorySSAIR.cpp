//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "MemorySSAIR.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

void pylir::MemSSA::MemorySSADialect::initialize() {
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

mlir::SuccessorOperands
pylir::MemSSA::MemoryBranchOp::getSuccessorOperands(unsigned index) {
  return mlir::SuccessorOperands(getBranchArgsMutable()[index]);
}

mlir::LogicalResult pylir::MemSSA::MemoryBranchOp::verify() {
  if (getBranchArgs().size() != getSuccessors().size()) {
    return emitOpError("Expected branch arguments for every successor");
  }
  return mlir::success();
}

void pylir::MemSSA::InstructionAttr::print(::mlir::AsmPrinter& printer) const {
  printer.getStream() << "// ";
  if (getInstruction()->getNumRegions() != 0) {
    printer.getStream()
        << getInstruction()->getRegisteredInfo()->getStringRef();
    return;
  }
  getInstruction()->print(printer.getStream(),
                          mlir::OpPrintingFlags{}.useLocalScope());
}

mlir::Attribute pylir::MemSSA::InstructionAttr::parse(::mlir::AsmParser& parser,
                                                      ::mlir::Type) {
  parser.emitError(parser.getCurrentLocation(),
                   "Parsing " + getMnemonic() + " not supported");
  return {};
}

void pylir::MemSSA::ReadWriteAttr::print(::mlir::AsmPrinter&) const {}

mlir::Attribute pylir::MemSSA::ReadWriteAttr::parse(::mlir::AsmParser& parser,
                                                    ::mlir::Type) {
  parser.emitError(parser.getCurrentLocation(),
                   "Parsing " + getMnemonic() + " not supported");
  return {};
}

mlir::RegionKind pylir::MemSSA::MemoryModuleOp::getRegionKind(unsigned int) {
  return mlir::RegionKind::SSACFG;
}

llvm::StringRef pylir::MemSSA::MemoryModuleOp::getDefaultDialect() {
  return pylir::MemSSA::MemorySSADialect::getDialectNamespace();
}

// TODO: I don't like this due to a fear of ODR if LLVM ever adds this. Might
// also want to upstream it myself.
namespace llvm {
template <class... Args>
llvm::hash_code hash_value(llvm::PointerUnion<Args...> ptr) {
  return llvm::hash_value(ptr.getOpaqueValue());
}
} // namespace llvm

#define GET_OP_CLASSES
#include "pylir/Optimizer/Analysis/MemorySSAIROps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/Analysis/MemorySSAIRTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/Analysis/MemorySSAIRAttributes.cpp.inc"
