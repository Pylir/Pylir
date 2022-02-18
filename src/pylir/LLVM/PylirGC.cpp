#include "PylirGC.hpp"

#include <llvm/CodeGen/AsmPrinter.h>
#include <llvm/CodeGen/GCMetadataPrinter.h>
#include <llvm/CodeGen/StackMaps.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCObjectFileInfo.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/Target/TargetLoweringObjectFile.h>
#include <llvm/Target/TargetMachine.h>

#include <pylir/Support/Macros.hpp>

namespace
{
class PylirGCStrategy final : public llvm::GCStrategy
{
public:
    PylirGCStrategy()
    {
        UseStatepoints = true;
        UsesMetadata = true;
    }

    llvm::Optional<bool> isGCManagedPointer(const llvm::Type* Ty) const override
    {
        if (!Ty->isPointerTy())
        {
            return llvm::None;
        }
        // Keep in Sync with PylirMemToLLVMIR.cpp
        return Ty->getPointerAddressSpace() == 1;
    }
};

// NOLINTNEXTLINE(cert-err58-cpp)
llvm::GCRegistry::Add<PylirGCStrategy> X("pylir-gc", "Garbage collector in Pylir");

class PylirGCMetaDataPrinter final : public llvm::GCMetadataPrinter
{
    void switchToPointerAlignedReadOnly(llvm::MCStreamer& os, llvm::AsmPrinter& printer)
    {
        llvm::SectionKind kind{};
        switch (printer.TM.getRelocationModel())
        {
            case llvm::Reloc::Static:
            case llvm::Reloc::ROPI:
            case llvm::Reloc::RWPI:
            case llvm::Reloc::ROPI_RWPI: kind = llvm::SectionKind::getReadOnly(); break;
            default: kind = llvm::SectionKind::getReadOnlyWithRel(); break;
        }
        auto alignment = printer.getDataLayout().getPointerABIAlignment(0);
        os.SwitchSection(
            printer.getObjFileLowering().getSectionForConstant(printer.getDataLayout(), kind, nullptr, alignment));
        os.emitValueToAlignment(alignment.value());
    }

    void writeStackMap(llvm::StackMaps& stackMaps, llvm::AsmPrinter& printer)
    {
        llvm::MCContext& context = printer.OutContext;
        auto& os = *printer.OutStreamer;

        auto* symbol = context.getOrCreateSymbol("pylir_stack_map");
        os.emitSymbolAttribute(symbol, llvm::MCSA_Global);
        switchToPointerAlignedReadOnly(os, printer);
        os.emitLabel(symbol);
        os.emitInt32(0x50594C52);

        struct CallSiteInfo
        {
            const llvm::MCExpr* programCounter;
            llvm::SmallVector<llvm::StackMaps::Location> locations;

            CallSiteInfo(const llvm::MCExpr* programCounter, llvm::ArrayRef<llvm::StackMaps::Location> locations)
                : programCounter(programCounter)
            {
                this->locations.reserve(locations.size());
                std::copy_if(locations.begin(), locations.end(), std::back_inserter(this->locations),
                             [](const llvm::StackMaps::Location& location)
                             {
                                 PYLIR_ASSERT(location.Type != llvm::StackMaps::Location::Unprocessed);
                                 return location.Type != llvm::StackMaps::Location::Constant
                                        && location.Type != llvm::StackMaps::Location::ConstantIndex;
                             });
                std::sort(this->locations.begin(), this->locations.end(),
                          [](const llvm::StackMaps::Location& lhs, const llvm::StackMaps::Location& rhs) {
                              return std::tie(lhs.Type, lhs.Size, lhs.Reg, lhs.Offset)
                                     < std::tie(rhs.Type, rhs.Size, rhs.Reg, rhs.Offset);
                          });
                this->locations.erase(
                    std::unique(this->locations.begin(), this->locations.end(),
                                [](const llvm::StackMaps::Location& lhs, const llvm::StackMaps::Location& rhs) {
                                    return std::tie(lhs.Type, lhs.Size, lhs.Reg, lhs.Offset)
                                           == std::tie(rhs.Type, rhs.Size, rhs.Reg, rhs.Offset);
                                }),
                    this->locations.end());
            }
        };

        std::vector<CallSiteInfo> callSiteInfos;
        auto currentFunction = stackMaps.getFnInfos().begin();
        std::size_t recordCount = 0;
        for (auto& iter : stackMaps.getCSInfos())
        {
            const auto* programCounter = llvm::MCBinaryExpr::createAdd(
                llvm::MCSymbolRefExpr::create(currentFunction->first, context), iter.CSOffsetExpr, context);

            if (++recordCount == currentFunction->second.RecordCount)
            {
                currentFunction++;
                recordCount = 0;
            }

            PYLIR_ASSERT(iter.Locations.size() >= 3);
            auto ref = llvm::makeArrayRef(iter.Locations).drop_front(3);
            if (ref.empty())
            {
                continue;
            }
            callSiteInfos.emplace_back(programCounter, ref);
        }

        auto pointerSize = printer.getDataLayout().getPointerSize();

        PYLIR_ASSERT(callSiteInfos.size() <= std::numeric_limits<std::uint32_t>::max());
        os.emitInt32(callSiteInfos.size());

        for (auto& iter : callSiteInfos)
        {
            os.emitValue(iter.programCounter, pointerSize);
            PYLIR_ASSERT(iter.locations.size() <= std::numeric_limits<std::uint32_t>::max());
            os.emitInt32(iter.locations.size());
            for (const auto& location : iter.locations)
            {
                PYLIR_ASSERT(location.Size == pointerSize);
                os.emitInt8(location.Type);
                os.emitInt8(0); // padding
                os.emitInt16(location.Reg);
                os.emitInt32(location.Offset);
            }
            os.emitInt32(0); // padding
        }
    }

    void writeGlobalMap(llvm::AsmPrinter& printer)
    {
        llvm::MCContext& context = printer.OutContext;

        std::vector<llvm::MCSymbol*> roots;
        std::vector<llvm::MCSymbol*> constants;
        std::vector<llvm::MCSymbol*> collections;
        for (const auto& iter : context.getSymbols())
        {
            auto* symbol = iter.getValue();
            if (!symbol->isInSection())
            {
                continue;
            }
            auto& section = symbol->getSection();
            auto name = section.getName();
            if (name == "py_root")
            {
                roots.push_back(symbol);
            }
            else if (name == "py_const")
            {
                constants.push_back(symbol);
            }
            else if (name == "py_coll")
            {
                collections.push_back(symbol);
            }
        }

        auto pointerSize = printer.getDataLayout().getPointerSize();

        auto& os = *printer.OutStreamer;
        switchToPointerAlignedReadOnly(os, printer);

        auto emitMap = [&](const std::vector<llvm::MCSymbol*>& values, llvm::Twine name)
        {
            auto* symbol = context.getOrCreateSymbol("pylir$" + name);
            os.emitSymbolAttribute(symbol, llvm::MCSA_Internal);
            os.emitLabel(symbol);
            for (const auto& iter : values)
            {
                os.emitSymbolValue(iter, pointerSize);
            }
            {
                auto* rootsStart = context.getOrCreateSymbol("pylir_" + name + "_start");
                os.emitSymbolAttribute(rootsStart, llvm::MCSA_Global);
                os.emitLabel(rootsStart);
                os.emitSymbolValue(symbol, pointerSize);
            }
            {
                auto* rootsEnd = context.getOrCreateSymbol("pylir_" + name + "_end");
                os.emitSymbolAttribute(rootsEnd, llvm::MCSA_Global);
                os.emitLabel(rootsEnd);
                os.emitValue(llvm::MCBinaryExpr::createAdd(
                                 llvm::MCSymbolRefExpr::create(symbol, context),
                                 llvm::MCConstantExpr::create(pointerSize * values.size(), context), context),
                             pointerSize);
            }
        };

        emitMap(roots, "roots");
        emitMap(constants, "constants");
        emitMap(collections, "collections");
    }

public:
    bool emitStackMaps(llvm::StackMaps& stackMaps, llvm::AsmPrinter& printer) override
    {
        writeStackMap(stackMaps, printer);
        writeGlobalMap(printer);
        return true;
    }
};

// NOLINTNEXTLINE(cert-err58-cpp)
llvm::GCMetadataPrinterRegistry::Add<PylirGCMetaDataPrinter> Y("pylir-gc", "Pylir GC stackmap writer");

} // namespace

void pylir::linkInGCStrategy() {}
