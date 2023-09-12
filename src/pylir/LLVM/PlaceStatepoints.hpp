//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/PassManager.h>

namespace pylir {
class PlaceStatepointsPass : public llvm::PassInfoMixin<PlaceStatepointsPass> {
public:
  explicit PlaceStatepointsPass() = default;

  llvm::PreservedAnalyses run(llvm::Module& M, llvm::ModuleAnalysisManager& AM);

  bool runOnFunction(llvm::Function& F, llvm::DominatorTree& DT,
                     llvm::TargetTransformInfo& TTI,
                     const llvm::TargetLibraryInfo& TLI);
};
} // namespace pylir
