//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Optimizer.hpp"

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirHIR/Transforms/Passes.hpp>
#include <pylir/Optimizer/PylirMem/Transforms/Passes.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Passes.hpp>
#include <pylir/Optimizer/Transforms/Passes.hpp>

static std::unique_ptr<mlir::Pass> createCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  // This is arguably not a canonicalization, not always beneficial and
  // expensive.
  config.enableRegionSimplification = false;
  return createCanonicalizerPass(config);
}

void pylir::registerOptimizationPipelines() {
  mlir::PassPipelineRegistration<>(
      "pylir-minimum",
      "Minimum pass pipeline to fully lower 'py' dialect, up until (exclusive) "
      "conversion to LLVM",
      [](mlir::OpPassManager& pm) {
        mlir::OpPassManager* nested = &pm.nestAny();
        nested->addPass(pylir::createDeadCodeEliminationPass());
        nested->addPass(HIR::createClassBodyOutliningPass());
        pm.addPass(HIR::createFuncOutliningPass());
        nested = &pm.nestAny();
        nested->addPass(pylir::createDeadCodeEliminationPass());
        pm.addPass(createConvertPylirHIRToPylirPyPass());

        nested = &pm.nestAny();
        nested->addPass(pylir::createDeadCodeEliminationPass());
        nested->addPass(Py::createExpandPyDialectPass());
        nested->addPass(pylir::createDeadCodeEliminationPass());
        pm.addPass(createConvertPylirPyToPylirMemPass());
      });

  mlir::PassPipelineRegistration<>(
      "pylir-optimize",
      "Optimization pipeline used by the compiler with lowering up until "
      "(exclusive) conversion to LLVM",
      [](mlir::OpPassManager& pm) {
        mlir::OpPassManager* nested = &pm.nestAny();
        nested->addPass(createCanonicalizerPass());
        nested->addPass(createDeadCodeEliminationPass());
        pm.addPass(Py::createGlobalSROAPass());

        pm.addPass(HIR::createClassBodyOutliningPass());
        pm.addPass(HIR::createFuncOutliningPass());
        nested = &pm.nestAny();
        nested->addPass(createCanonicalizerPass());
        nested->addPass(createDeadCodeEliminationPass());
        pm.addPass(createConvertPylirHIRToPylirPyPass());

        mlir::OpPassManager inlinerNested;

        inlinerNested.addPass(createCanonicalizerPass());
        inlinerNested.nestAny().addPass(
            Py::createGlobalLoadStoreEliminationPass());
        inlinerNested.addPass(Py::createFoldGlobalsPass());
        inlinerNested.addPass(mlir::createSymbolDCEPass());
        nested = &inlinerNested.nestAny();
        nested->addPass(createCanonicalizerPass());
        nested->addPass(mlir::createCSEPass());
        nested->addPass(pylir::createConditionalsImplicationsPass());
        nested->addPass(createCanonicalizerPass());
        nested->addPass(createLoadForwardingPass());

        // TODO: Upstream MLIR has a bug making SCCP that is not module
        //  level not thread-safe. This is caught by TSAN.
        inlinerNested.addPass(mlir::createSCCPPass());
        nested = &inlinerNested.nestAny();
        nested->addPass(createDeadCodeEliminationPass());
        nested->addPass(Py::createExpandPyDialectPass());
        nested->addPass(createCanonicalizerPass());
        nested->addPass(mlir::createCSEPass());
        nested->addPass(createLoadForwardingPass());
        inlinerNested.addPass(mlir::createSCCPPass());
        nested = &inlinerNested.nestAny();
        nested->addPass(createCanonicalizerPass());

        Py::InlinerPassOptions options{};
        std::string pipeline;
        llvm::raw_string_ostream ss(pipeline);
        inlinerNested.printAsTextualPipeline(ss);
        options.m_optimizationPipeline = std::move(pipeline);
        pm.addPass(Py::createInlinerPass(options));
        pm.nestAny().addPass(createDeadCodeEliminationPass());
        pm.addPass(createConvertPylirPyToPylirMemPass());
      });

  mlir::PassPipelineRegistration<PylirLLVMOptions>(
      "pylir-llvm",
      "Pass pipeline used to lower 'pylir-minimum' and 'pylir-optimize' output "
      "to LLVM",
      [](mlir::OpPassManager& pm, const PylirLLVMOptions& options) {
        auto* nested = &pm.nestAny();
        nested->addPass(pylir::createDeadCodeEliminationPass());
        nested->addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(createConvertPylirToLLVMPass(ConvertPylirToLLVMPassOptions{
            options.targetTriple, options.dataLayout}));
        nested = &pm.nestAny();
        nested->addPass(mlir::createReconcileUnrealizedCastsPass());
        // nested->addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
        nested->addPass(mlir::LLVM::createLegalizeForExportPass());
      });
}
