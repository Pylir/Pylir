#pragma once

#include <llvm/IR/PassManager.h>

namespace pylir
{
class PlaceStatepointsPass : public llvm::PassInfoMixin<PlaceStatepointsPass>
{
public:
    explicit PlaceStatepointsPass() = default;

    llvm::PreservedAnalyses run(llvm::Function& function, llvm::FunctionAnalysisManager& am);
};
} // namespace pylir
