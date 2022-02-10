#include "PylirGC.hpp"

#include <llvm/IR/DerivedTypes.h>

namespace
{
class PylirGCStrategy : public llvm::GCStrategy
{
public:
    PylirGCStrategy()
    {
        UseStatepoints = true;
        // TODO: UsesMetadata = true;
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

} // namespace

void pylir::linkInGCStrategy() {}
