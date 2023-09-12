//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Pass/PassManager.h>

#include <llvm/Support/BLAKE3.h>

#include "Passes.hpp"

namespace pylir {
#define GEN_PASS_DEF_FIXPOINTPASS
#include "pylir/Optimizer/Transforms/Passes.h.inc"
} // namespace pylir

namespace {
class FixpointPass : public pylir::impl::FixpointPassBase<FixpointPass> {
  mlir::OpPassManager m_passManager;

  void runOnOperation() override;

  mlir::LogicalResult initialize(mlir::MLIRContext*) override {
    return mlir::parsePassPipeline(m_optimizationPipeline, m_passManager);
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    Base::getDependentDialects(registry);
    // Above initialize will signal the error properly. This also gets called
    // before `initialize`, hence we can't use m_passManager here.
    mlir::OpPassManager temp;
    if (mlir::failed(mlir::parsePassPipeline(m_optimizationPipeline, temp,
                                             llvm::nulls())))
      return;

    temp.getDependentDialects(registry);
  }

  llvm::BLAKE3Result<> getFingerprint() {
    llvm::BLAKE3 hasher;

    auto addToHash = [&](const auto& data) {
      hasher.update(llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t*>(&data),
          sizeof(std::remove_reference_t<
                 decltype(data)>))); // NOLINT(bugprone-sizeof-expression)
    };

    getOperation()->walk([&](mlir::Operation* op) {
      addToHash(op);
      addToHash(op->hashProperties());
      addToHash(op->getDiscardableAttrDictionary());
      for (mlir::Region& region : op->getRegions()) {
        for (mlir::Block& block : region) {
          addToHash(&block);
          for (mlir::BlockArgument& arg : block.getArguments())
            addToHash(arg.getAsOpaquePointer());
        }
      }
      addToHash(op->getLoc().getAsOpaquePointer());
      for (mlir::Value operand : op->getOperands())
        addToHash(operand.getAsOpaquePointer());

      for (mlir::Block* block : op->getSuccessors())
        addToHash(block);
    });
    return hasher.result();
  }

public:
  using Base::Base;
};

void FixpointPass::runOnOperation() {
  auto startFingerprint = getFingerprint();
  for (std::size_t i = 0; i < m_maxIterationCount; i++) {
    if (mlir::failed(runPipeline(m_passManager, getOperation()))) {
      signalPassFailure();
      return;
    }
    auto endFingerPrint = getFingerprint();
    if (endFingerPrint == startFingerprint)
      return;

    startFingerprint = endFingerPrint;
  }
  m_maxIterationReached++;
}

} // namespace
