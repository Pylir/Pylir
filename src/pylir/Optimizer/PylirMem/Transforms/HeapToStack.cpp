// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/DialectRegistry.h>

#include <pylir/Optimizer/Analysis/EscapeAnalysis.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemOps.hpp>
#include <pylir/Optimizer/PylirMem/IR/Value.hpp>

#include "Passes.hpp"

namespace pylir::Mem {
#define GEN_PASS_DEF_HEAPTOSTACKPASS
#include "pylir/Optimizer/PylirMem/Transforms/Passes.h.inc"
} // namespace pylir::Mem

namespace {
class HeapToStack : public pylir::Mem::impl::HeapToStackPassBase<HeapToStack> {
public:
  using Base::Base;

protected:
  void runOnOperation() override {
    auto& escapeAnalysis = getAnalysis<pylir::EscapeAnalysis>();

    llvm::DenseMap<mlir::Attribute, pylir::Mem::LayoutType> cache;
    getOperation()->walk([&](pylir::Mem::GCAllocObjectOp allocObjectOp) {
      llvm::APInt trailingObjects;
      if (!mlir::matchPattern(allocObjectOp.getTrailingItems(),
                              mlir::m_ConstantInt(&trailingObjects))) {
        return;
      }
      if (trailingObjects.ugt(m_maxObjectSize)) {
        return;
      }

      auto layoutType =
          pylir::Mem::getLayoutType(allocObjectOp.getTypeObject(), &cache);
      if (!layoutType) {
        return;
      }

      for (mlir::Operation* iter : allocObjectOp->getUsers()) {
        PYLIR_ASSERT(iter->getDialect()->getNamespace() == "pyMem" &&
                     "Can only deal with 'pyMem' operations");
        PYLIR_ASSERT(iter->getNumResults() == 1 &&
                     "Expected only a single result");
        if (escapeAnalysis.escapes(iter->getResult(0))) {
          return;
        }
      }

      mlir::OpBuilder builder(allocObjectOp);
      auto stackAlloc = builder.create<pylir::Mem::StackAllocObjectOp>(
          allocObjectOp->getLoc(), allocObjectOp.getTypeObject(), *layoutType,
          trailingObjects);
      allocObjectOp.replaceAllUsesWith(static_cast<mlir::Value>(stackAlloc));
      m_heapAllocationsReplaced++;
    });
  }
};
} // namespace
