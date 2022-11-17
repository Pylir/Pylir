//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Optimizer.hpp"

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirMem/Transforms/Passes.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Passes.hpp>
#include <pylir/Optimizer/Transforms/Passes.hpp>

void pylir::registerOptimizationPipelines()
{
    mlir::PassPipelineRegistration<>(
        "pylir-minimum", "Minimum pass pipeline to fully lower 'py' dialect, up until (exclusive) conversion to LLVM",
        [](mlir::OpPassManager& pm)
        {
            // This is supposed to be the minimum pipeline, so shouldn't really contain the canonicalizations, but the
            // dialect conversion framework currently cannot deal with statically known dead code.
            // Running the canonicalizer eliminates any such occurrences.
            mlir::OpPassManager* nested = &pm.nestAny();
            nested->addPass(mlir::createCanonicalizerPass());
            nested->addPass(Py::createExpandPyDialectPass());
            nested->addPass(mlir::createCanonicalizerPass());
            pm.addPass(createConvertPylirPyToPylirMemPass());
        });

    mlir::PassPipelineRegistration<>(
        "pylir-optimize",
        "Optimization pipeline used by the compiler with lowering up until (exclusive) conversion to LLVM",
        [](mlir::OpPassManager& pm)
        {
            mlir::OpPassManager* nested;
            pm.addPass(mlir::createCanonicalizerPass());
            pm.nestAny().addPass(Py::createGlobalLoadStoreEliminationPass());
            pm.addPass(Py::createFoldGlobalsPass());
            pm.nestAny().addPass(mlir::createCSEPass());
            pm.addPass(Py::createTrialInlinerPass());
            pm.addPass(mlir::createSymbolDCEPass());
            nested = &pm.nestAny();
            nested->addPass(createLoadForwardingPass());
            nested->addPass(mlir::createSCCPPass());
            pm.addPass(Py::createMonomorphPass());
            pm.addPass(Py::createTrialInlinerPass());
            pm.addPass(mlir::createSymbolDCEPass());
            nested = &pm.nestAny();
            nested->addPass(Py::createExpandPyDialectPass());
            nested->addPass(mlir::createCanonicalizerPass());
            nested->addPass(mlir::createCSEPass());
            nested->addPass(createLoadForwardingPass());
            nested->addPass(mlir::createSCCPPass());
            nested->addPass(mlir::createCanonicalizerPass());
            pm.addPass(createConvertPylirPyToPylirMemPass());
            nested = &pm.nestAny();
            nested->addPass(mlir::createCanonicalizerPass());
            nested->addPass(Mem::createHeapToStackPass());
        });

    struct PylirLLVMOptions : public mlir::PassPipelineOptions<PylirLLVMOptions>
    {
        Option<std::string> targetTriple{*this, "target-triple", llvm::cl::desc("LLVM target triple"),
                                         llvm::cl::init(LLVM_DEFAULT_TARGET_TRIPLE)};
        Option<std::string> dataLayout{*this, "data-layout", llvm::cl::desc("LLVM data layout"), llvm::cl::init("")};
    };

    mlir::PassPipelineRegistration<PylirLLVMOptions>(
        "pylir-llvm", "Pass pipeline used to lower 'pylir-minimum' and 'pylir-optimize' output to LLVM",
        [](mlir::OpPassManager& pm, const PylirLLVMOptions& options)
        {
            auto* nested = &pm.nestAny();
            nested->addPass(mlir::arith::createArithExpandOpsPass());
            nested->addPass(mlir::createArithToLLVMConversionPass());
            pm.addPass(
                createConvertPylirToLLVMPass(ConvertPylirToLLVMPassOptions{options.targetTriple, options.dataLayout}));
            nested = &pm.nestAny();
            nested->addPass(mlir::createReconcileUnrealizedCastsPass());
            nested->addPass(mlir::LLVM::createLegalizeForExportPass());
        });
}