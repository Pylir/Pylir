#include "PylirGC.hpp"

#include <llvm/CodeGen/AsmPrinter.h>
#include <llvm/CodeGen/GCMetadataPrinter.h>
#include <llvm/CodeGen/StackMaps.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCObjectFileInfo.h>
#include <llvm/MC/MCStreamer.h>

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
public:
    bool emitStackMaps(llvm::StackMaps& SM, llvm::AsmPrinter& AP) override
    {
        llvm::MCContext& context = AP.OutContext;
        auto& os = *AP.OutStreamer;

        auto* symbol = context.getOrCreateSymbol("pylir_stack_map");
        os.emitSymbolAttribute(symbol, llvm::MCSA_Global);
        os.SwitchSection(context.getObjectFileInfo()->getReadOnlySection());
        os.emitValueToAlignment(8);
        os.emitLabel(symbol);
        os.emitInt32(0x50594C52);

        struct CallSiteInfo
        {
            const llvm::MCExpr* programCounter;
            llvm::SmallVector<llvm::StackMaps::Location> locations;

            CallSiteInfo(const llvm::MCExpr* programCounter, llvm::ArrayRef<llvm::StackMaps::Location> locations)
                : programCounter(programCounter), locations(locations.begin(), locations.end())
            {
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
        auto currentFunction = SM.getFnInfos().begin();
        std::size_t recordCount = 0;
        for (auto& iter : SM.getCSInfos())
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

        PYLIR_ASSERT(callSiteInfos.size() <= std::numeric_limits<std::uint32_t>::max());
        os.emitInt32(callSiteInfos.size());

        for (auto& iter : callSiteInfos)
        {
            os.emitValue(iter.programCounter, 8);
            PYLIR_ASSERT(iter.locations.size() <= std::numeric_limits<std::uint32_t>::max());
            os.emitInt32(iter.locations.size());
            for (const auto& location : iter.locations)
            {
                PYLIR_ASSERT(location.Type == llvm::StackMaps::Location::Direct
                             || location.Type == llvm::StackMaps::Location::Indirect);
                PYLIR_ASSERT(location.Size == 8);
                os.emitInt8(location.Type);
                os.emitInt8(0); // padding
                os.emitInt16(location.Reg);
                os.emitInt32(location.Offset);
            }
            os.emitInt32(0); // padding
        }

        return true;
    }
};

// NOLINTNEXTLINE(cert-err58-cpp)
llvm::GCMetadataPrinterRegistry::Add<PylirGCMetaDataPrinter> Y("pylir-gc", "Pylir GC stackmap writer");

} // namespace

void pylir::linkInGCStrategy() {}
