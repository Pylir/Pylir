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
#include <mlir/Transforms/Passes.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirHIR/Transforms/Passes.hpp>
#include <pylir/Optimizer/PylirMem/Transforms/Passes.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Passes.hpp>
#include <pylir/Optimizer/Transforms/Passes.hpp>

void pylir::registerOptimizationPipelines() {
  mlir::PassPipelineRegistration<>(
      "pylir-minimum",
      "Minimum pass pipeline to fully lower 'py' dialect, up until (exclusive) "
      "conversion to LLVM",
      [](mlir::OpPassManager& pm) {
        pm.addPass(HIR::createFuncOutliningPass());
        mlir::OpPassManager* nested = &pm.nestAny();
        // This is supposed to be the minimum pipeline, so shouldn't really
        // contain the canonicalizations, but the dialect conversion framework
        // currently cannot deal with statically known dead code. Running the
        // canonicalizer eliminates any such occurrences.
        nested->addPass(mlir::createCanonicalizerPass());
        pm.addPass(createConvertPylirHIRToPylirPyPass());

        nested = &pm.nestAny();
        nested->addPass(mlir::createCanonicalizerPass());
        nested->addPass(Py::createExpandPyDialectPass());
        nested->addPass(mlir::createCanonicalizerPass());
        pm.addPass(createConvertPylirPyToPylirMemPass());
      });

  mlir::PassPipelineRegistration<>(
      "pylir-optimize",
      "Optimization pipeline used by the compiler with lowering up until "
      "(exclusive) conversion to LLVM",
      [](mlir::OpPassManager& pm) {
        pm.addPass(HIR::createFuncOutliningPass());
        pm.nestAny().addPass(mlir::createCanonicalizerPass());
        pm.addPass(createConvertPylirHIRToPylirPyPass());

        mlir::OpPassManager inlinerNested;

        mlir::OpPassManager* nested;
        inlinerNested.addPass(mlir::createCanonicalizerPass());
        inlinerNested.nestAny().addPass(
            Py::createGlobalLoadStoreEliminationPass());
        inlinerNested.addPass(Py::createFoldGlobalsPass());
        inlinerNested.addPass(mlir::createSymbolDCEPass());
        nested = &inlinerNested.nestAny();
        nested->addPass(mlir::createCanonicalizerPass());
        nested->addPass(mlir::createCSEPass());
        nested->addPass(pylir::createConditionalsImplicationsPass());
        nested->addPass(mlir::createCanonicalizerPass());
        nested->addPass(createLoadForwardingPass());

        // TODO: Upstream MLIR has a bug making SCCP that is not module
        //  level not thread-safe. This is caught by TSAN.
        inlinerNested.addPass(mlir::createSCCPPass());
        nested = &inlinerNested.nestAny();
        nested->addPass(Py::createExpandPyDialectPass());
        nested->addPass(mlir::createCanonicalizerPass());
        nested->addPass(mlir::createCSEPass());
        nested->addPass(createLoadForwardingPass());
        inlinerNested.addPass(mlir::createSCCPPass());
        nested = &inlinerNested.nestAny();
        nested->addPass(mlir::createCanonicalizerPass());

        Py::InlinerPassOptions options{};
        std::string pipeline;
        llvm::raw_string_ostream ss(pipeline);
        inlinerNested.printAsTextualPipeline(ss);
        options.m_optimizationPipeline = std::move(pipeline);
        pm.addPass(Py::createInlinerPass(options));

        pm.addPass(createConvertPylirPyToPylirMemPass());
        nested = &pm.nestAny();
        nested->addPass(mlir::createCanonicalizerPass());
        // TODO: Re-enable once it properly supports destructors.
        // nested->addPass(Mem::createHeapToStackPass());
      });

  mlir::PassPipelineRegistration<PylirLLVMOptions>(
      "pylir-llvm",
      "Pass pipeline used to lower 'pylir-minimum' and 'pylir-optimize' output "
      "to LLVM",
      [](mlir::OpPassManager& pm, const PylirLLVMOptions& options) {
        auto* nested = &pm.nestAny();
        nested->addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(createConvertPylirToLLVMPass(ConvertPylirToLLVMPassOptions{
            options.targetTriple, options.dataLayout}));
        nested = &pm.nestAny();
        nested->addPass(mlir::createReconcileUnrealizedCastsPass());
        // nested->addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
        nested->addPass(mlir::LLVM::createLegalizeForExportPass());
      });
}
