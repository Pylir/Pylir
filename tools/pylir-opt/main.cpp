//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/AsmState.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/DebugCounter.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/Timing.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/ToolOutputFile.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Passes.hpp>
#include <pylir/Optimizer/Transforms/Passes.hpp>

#include "Passes.hpp"
#include "TestDialect.hpp"

int main(int argc, char** argv)
{
    mlir::registerAllPasses();

    mlir::DialectRegistry registry;
    registry.insert<pylir::Mem::PylirMemDialect, pylir::Py::PylirPyDialect, pylir::test::TestDialect>();
    mlir::registerAllDialects(registry);

    pylir::registerConversionPasses();
    pylir::registerTransformPasses();
    pylir::Py::registerTransformPasses();
    pylir::test::registerTestPasses();

    // Reimplementation of MlirOptMain overload that sets up all the command line options. The purpose of doing so is
    // to implicitly run `pylir-finalize-ref-attrs` pass over the whole module before any other compiler passes are run.
    // TODO: Consider adding an overload upstream that allows us to customize the pass pipeline without having to
    //       reimplement all of main.

    static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                                                    llvm::cl::init("-"));

    static llvm::cl::opt<std::string> outputFilename("o", llvm::cl::desc("Output filename"),
                                                     llvm::cl::value_desc("filename"), llvm::cl::init("-"));

    static llvm::cl::opt<bool> splitInputFile("split-input-file",
                                              llvm::cl::desc("Split the input file into pieces and process each "
                                                             "chunk independently"),
                                              llvm::cl::init(false));

    static llvm::cl::opt<bool> verifyDiagnostics("verify-diagnostics",
                                                 llvm::cl::desc("Check that emitted diagnostics match "
                                                                "expected-* lines on the corresponding line"),
                                                 llvm::cl::init(false));

    static llvm::cl::opt<bool> verifyPasses(
        "verify-each", llvm::cl::desc("Run the verifier after each transformation pass"), llvm::cl::init(true));

    static llvm::cl::opt<bool> allowUnregisteredDialects("allow-unregistered-dialect",
                                                         llvm::cl::desc("Allow operation with no registered dialects"),
                                                         llvm::cl::init(false));

    static llvm::cl::opt<bool> showDialects("show-dialects", llvm::cl::desc("Print the list of registered dialects"),
                                            llvm::cl::init(false));

    static llvm::cl::opt<bool> emitBytecode("emit-bytecode", llvm::cl::desc("Emit bytecode when generating output"),
                                            llvm::cl::init(false));

    llvm::InitLLVM y(argc, argv);

    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    mlir::registerDefaultTimingManagerCLOptions();
    mlir::DebugCounter::registerCLOptions();
    mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");

    // Build the list of dialects as a header for the --help message.
    std::string helpHeader = "pylir-opt\nAvailable Dialects: ";
    {
        llvm::raw_string_ostream os(helpHeader);
        interleaveComma(registry.getDialectNames(), os, [&](auto name) { os << name; });
    }
    // Parse pass names in main to ensure static initialization completed.
    llvm::cl::ParseCommandLineOptions(argc, argv, helpHeader);

    if (showDialects)
    {
        llvm::outs() << "Available Dialects:\n";
        interleave(
            registry.getDialectNames(), llvm::outs(), [](auto name) { llvm::outs() << name; }, "\n");
        return -1;
    }

    // Set up the input file.
    std::string errorMessage;
    auto file = mlir::openInputFile(inputFilename, &errorMessage);
    if (!file)
    {
        llvm::errs() << errorMessage << "\n";
        return -1;
    }

    auto output = mlir::openOutputFile(outputFilename, &errorMessage);
    if (!output)
    {
        llvm::errs() << errorMessage << "\n";
        return -1;
    }

    if (failed(MlirOptMain(
            output->os(), std::move(file),
            [&](mlir::PassManager& pm)
            {
                pm.addPass(pylir::Py::createFinalizeRefAttrsPass());
                return passPipeline.addToPipeline(pm,
                                                  [&](const llvm::Twine& msg)
                                                  {
                                                      mlir::emitError(mlir::UnknownLoc::get(pm.getContext())) << msg;
                                                      return mlir::failure();
                                                  });
            },
            registry, splitInputFile, verifyDiagnostics, verifyPasses, allowUnregisteredDialects, false, emitBytecode)))
    {
        return -1;
    }

    // Keep the output file if the invocation of MlirOptMain was successful.
    output->keep();
}
