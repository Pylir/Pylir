//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PylirGC.hpp"

#include <llvm/ADT/SetVector.h>
#include <llvm/CodeGen/AsmPrinter.h>
#include <llvm/CodeGen/GCMetadataPrinter.h>
#include <llvm/CodeGen/StackMaps.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCObjectFileInfo.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCValue.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Target/TargetLoweringObjectFile.h>
#include <llvm/Target/TargetMachine.h>

#include <pylir/Support/Macros.hpp>

namespace {

// NOLINTNEXTLINE(cert-err58-cpp)
llvm::cl::opt<bool> emitStackMap("pylir-emit-stackmap", llvm::cl::Hidden,
                                 llvm::cl::init(true));

class PylirGCStrategy final : public llvm::GCStrategy {
public:
  PylirGCStrategy() {
    UseStatepoints = true;
    UsesMetadata = true;
  }

  std::optional<bool> isGCManagedPointer(const llvm::Type* Ty) const override {
    if (!Ty->isPointerTy())
      return std::nullopt;

    // Keep in Sync with PylirMemToLLVMIR.cpp
    return Ty->getPointerAddressSpace() == 1;
  }
};

// NOLINTNEXTLINE(cert-err58-cpp)
llvm::GCRegistry::Add<PylirGCStrategy> x("pylir-gc",
                                         "Garbage collector in Pylir");

class PylirGCMetaDataPrinter final : public llvm::GCMetadataPrinter {
  void switchToPointerAlignedReadOnly(llvm::MCStreamer& os,
                                      llvm::AsmPrinter& printer) {
    llvm::SectionKind kind{};
    switch (printer.TM.getRelocationModel()) {
    case llvm::Reloc::Static:
    case llvm::Reloc::ROPI:
    case llvm::Reloc::RWPI:
    case llvm::Reloc::ROPI_RWPI: kind = llvm::SectionKind::getReadOnly(); break;
    default: kind = llvm::SectionKind::getReadOnlyWithRel(); break;
    }
    auto alignment = printer.getDataLayout().getPointerABIAlignment(0);
    os.switchSection(printer.getObjFileLowering().getSectionForConstant(
        printer.getDataLayout(), kind, nullptr, alignment));
    os.emitValueToAlignment(alignment);
  }

  /// Writes out the stack map in our custom format. See pylir/Runtime/Stack.cpp
  /// for details of the format.
  void writeStackMap(llvm::StackMaps& stackMaps, llvm::AsmPrinter& printer) {
    llvm::MCContext& context = printer.OutContext;
    auto& os = *printer.OutStreamer;

    auto* symbol = printer.GetExternalSymbolSymbol("pylir_stack_map");
    os.emitSymbolAttribute(symbol, llvm::MCSA_Global);
    switchToPointerAlignedReadOnly(os, printer);
    os.emitLabel(symbol);
    // File magic 'PYLR' as uint32_t
    os.emitInt32(0x50594C52);

    auto stackMapLocComp = [](const llvm::StackMaps::Location& lhs,
                              const llvm::StackMaps::Location& rhs) {
      return std::tie(lhs.Type, lhs.Reg, lhs.Offset, lhs.Size) <
             std::tie(rhs.Type, rhs.Reg, rhs.Offset, rhs.Size);
    };

    auto locNoConstantPred = [](const llvm::StackMaps::Location& location) {
      return location.Type != llvm::StackMaps::Location::Constant &&
             location.Type != llvm::StackMaps::Location::ConstantIndex;
    };

    std::vector<llvm::StackMaps::Location> allLocations;
    for (llvm::StackMaps::CallsiteInfo& iter : stackMaps.getCSInfos()) {
      // First three entries always contain metadata about things like calling
      // convention used, that we just aren't interested in.
      auto ref = llvm::ArrayRef(iter.Locations).drop_front(3);

      llvm::copy_if(ref, std::back_inserter(allLocations), locNoConstantPred);
    }

    // There are a lot of locations across all call sites that are of the same
    // type, size, register, offset etc. We therefore unique them all, put them
    // in a single large array and have the callsites reference them via index.
    llvm::sort(allLocations, stackMapLocComp);
    allLocations.erase(llvm::unique(allLocations,
                                    [](const llvm::StackMaps::Location& lhs,
                                       const llvm::StackMaps::Location& rhs) {
                                      return std::tie(lhs.Type, lhs.Size,
                                                      lhs.Reg, lhs.Offset) ==
                                             std::tie(rhs.Type, rhs.Size,
                                                      rhs.Reg, rhs.Offset);
                                    }),
                       allLocations.end());
    allLocations.shrink_to_fit();

    auto getLocIndex = [&](const llvm::StackMaps::Location& location) {
      return llvm::lower_bound(allLocations, location, stackMapLocComp) -
             allLocations.begin();
    };

    struct CallSiteInfo {
      const llvm::MCExpr* programCounter;
      std::vector<std::uint32_t> locationIndices;
    };

    std::vector<CallSiteInfo> callSiteInfos;
    auto* currentFunction = stackMaps.getFnInfos().begin();
    std::size_t recordCount = 0;
    for (auto& iter : stackMaps.getCSInfos()) {
      const auto* programCounter = llvm::MCBinaryExpr::createAdd(
          llvm::MCSymbolRefExpr::create(currentFunction->first, context),
          iter.CSOffsetExpr, context);

      if (++recordCount == currentFunction->second.RecordCount) {
        currentFunction++;
        recordCount = 0;
      }

      std::vector<std::uint32_t> locIndices;
      llvm::transform(
          llvm::make_filter_range(iter.Locations, locNoConstantPred),
          std::back_inserter(locIndices), getLocIndex);
      if (locIndices.empty())
        continue;

      // Sort the indices for a better access pattern. This also has the nice
      // side effect of making similar location types be right after each other
      // as well. E.g. multiple 'Indirect' accesses right after each other. This
      // is due to 'allLocations' being sorted.
      llvm::sort(locIndices);
      locIndices.erase(llvm::unique(locIndices, std::equal_to<>{}),
                       locIndices.end());
      locIndices.shrink_to_fit();

      callSiteInfos.push_back({programCounter, std::move(locIndices)});
    }

    auto pointerSize = printer.getDataLayout().getPointerSize();

    os.emitULEB128IntValue(allLocations.size());
    for (auto& iter : allLocations) {
      PYLIR_ASSERT(iter.Size % pointerSize == 0 &&
                   "Expected only pointers (or a vector of) in stackmap entry");
      os.emitInt8(iter.Type);
      os.emitULEB128IntValue(iter.Reg);
      if (iter.Type != llvm::StackMaps::Location::Register)
        os.emitSLEB128IntValue(iter.Offset);

      if (iter.Type == llvm::StackMaps::Location::Indirect) {
        PYLIR_ASSERT(iter.Size / pointerSize <=
                     std::numeric_limits<std::uint8_t>::max());
        os.emitInt8(iter.Size / pointerSize);
      }
    }

    os.emitULEB128IntValue(callSiteInfos.size());
    for (auto& iter : callSiteInfos) {
      // Mach-O arm64, seem to require that relocations are placed with proper
      // alignment. Documentation is incredibly sparse however, and I fear this
      // might be a requirement on more platforms. For the time being, we'll
      // just require this everywhere. At worst, we are wasting pointerSize-1
      // bytes per call site.
      // TODO: Reference documentation for alignment requirement.
      os.emitValueToAlignment(llvm::Align(pointerSize));
      os.emitValue(iter.programCounter, pointerSize);
      os.emitULEB128IntValue(iter.locationIndices.size());
      for (std::uint32_t index : iter.locationIndices)
        os.emitULEB128IntValue(index);
    }
  }

  void writeGlobalMap(llvm::AsmPrinter& printer) {
    llvm::MCContext& context = printer.OutContext;

    auto symbolComp = [](llvm::MCSymbol* lhs, llvm::MCSymbol* rhs) {
      return lhs->getName() < rhs->getName();
    };

    std::vector<llvm::MCSymbol*> roots;
    std::vector<llvm::MCSymbol*> constants;
    std::vector<llvm::MCSymbol*> collections;
    for (const auto& iter : context.getSymbols()) {
      auto* symbol = iter.getValue();
      if (!symbol->isInSection())
        continue;

      auto& section = symbol->getSection();
      auto* container =
          llvm::StringSwitch<std::vector<llvm::MCSymbol*>*>(section.getName())
              .Case("py_root", &roots)
              .Case("py_const", &constants)
              .Case("py_coll", &collections)
              .Default(nullptr);
      if (!container)
        continue;

      auto pos = std::lower_bound(container->begin(), container->end(), symbol,
                                  symbolComp);
      container->insert(pos, symbol);
    }

    auto pointerSize = printer.getDataLayout().getPointerSize();

    auto& os = *printer.OutStreamer;
    switchToPointerAlignedReadOnly(os, printer);

    auto emitMap = [&](llvm::ArrayRef<llvm::MCSymbol*> values,
                       llvm::Twine name) {
      auto* symbol = printer.GetExternalSymbolSymbol(("pylir$" + name).str());
      os.emitSymbolAttribute(symbol, llvm::MCSA_Internal);
      os.emitLabel(symbol);
      for (const auto& iter : values)
        os.emitSymbolValue(iter, pointerSize);

      {
        auto* rootsStart =
            printer.GetExternalSymbolSymbol(("pylir_" + name + "_start").str());
        os.emitSymbolAttribute(rootsStart, llvm::MCSA_Global);
        os.emitLabel(rootsStart);
        os.emitSymbolValue(symbol, pointerSize);
      }
      {
        auto* rootsEnd =
            printer.GetExternalSymbolSymbol(("pylir_" + name + "_end").str());
        os.emitSymbolAttribute(rootsEnd, llvm::MCSA_Global);
        os.emitLabel(rootsEnd);
        os.emitValue(llvm::MCBinaryExpr::createAdd(
                         llvm::MCSymbolRefExpr::create(symbol, context),
                         llvm::MCConstantExpr::create(
                             pointerSize * values.size(), context),
                         context),
                     pointerSize);
      }
    };

    emitMap(roots, "roots");
    emitMap(constants, "constants");
    emitMap(collections, "collections");
  }

public:
  bool emitStackMaps(llvm::StackMaps& stackMaps,
                     llvm::AsmPrinter& printer) override {
    if (!emitStackMap) {
      // We claim to have handled the stack map emission anyway, since we
      // explicitly do not want one.
      return true;
    }
    writeStackMap(stackMaps, printer);
    writeGlobalMap(printer);
    return true;
  }
};

llvm::GCMetadataPrinterRegistry::Add<PylirGCMetaDataPrinter>
    // NOLINTNEXTLINE(cert-err58-cpp)
    y("pylir-gc", "Pylir GC stackmap writer");

} // namespace

void pylir::linkInGCStrategy() {}
