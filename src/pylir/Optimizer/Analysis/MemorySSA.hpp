//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/AnalysisManager.h>

#include <llvm/ADT/DenseMap.h>

#include <pylir/Optimizer/Transforms/Util/SSABuilder.hpp>
#include <pylir/Support/Macros.hpp>

#include <memory>

#include "MemorySSAIR.hpp"

namespace mlir {
class ImplicitLocOpBuilder;
} // namespace mlir

namespace pylir {

class MemorySSA {
  mlir::OwningOpRef<MemSSA::MemoryModuleOp> m_region;
  llvm::DenseMap<mlir::Block*, mlir::Block*> m_blockMapping;

  void createIR(mlir::Operation* operation);

  void optimizeUses(mlir::AnalysisManager& analysisManager);

  void fillRegion(mlir::Region& region, mlir::ImplicitLocOpBuilder& builder,
                  pylir::SSABuilder& ssaBuilder,
                  llvm::MapVector<mlir::SideEffects::Resource*,
                                  pylir::SSABuilder::DefinitionsMap>& lastDefs,
                  llvm::ArrayRef<mlir::Block*> regionSuccessors);

public:
  explicit MemorySSA(mlir::Operation* operation,
                     mlir::AnalysisManager& analysisManager);

  mlir::Region& getMemoryRegion() {
    return m_region->getBody();
  }

  void dump() const;

  void print(llvm::raw_ostream& out) const;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const MemorySSA& ssa) {
    ssa.print(os);
    return os;
  }
};

} // namespace pylir
