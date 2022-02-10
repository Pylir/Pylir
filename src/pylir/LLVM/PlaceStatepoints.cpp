#include "PlaceStatepoints.hpp"

llvm::PreservedAnalyses pylir::PlaceStatepointsPass::run(llvm::Function& function, llvm::FunctionAnalysisManager& am)
{
    if (function.isDeclaration() || function.empty())
    {
        return llvm::PreservedAnalyses::all();
    }

    return llvm::PreservedAnalyses::all();
}
