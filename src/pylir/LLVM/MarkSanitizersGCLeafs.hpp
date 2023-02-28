// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/PassManager.h>

namespace pylir
{
/// Convenience pass placed right after any sanitizer or similar instrumentation passes.
/// This pass does basically nothing but mark calls to instrumentation functions as "gc-leaf-functions" to improve
/// codegen of the output and avoid calls to these functions being converted to state points.
class MarkSanitizersGCLeafsPass : public llvm::PassInfoMixin<MarkSanitizersGCLeafsPass>
{
public:
    explicit MarkSanitizersGCLeafsPass() = default;

    /// Run function with signature indicating the pass manager that this is a module pass.
    llvm::PreservedAnalyses run(llvm::Module& M, llvm::ModuleAnalysisManager& AM);
};
} // namespace pylir
