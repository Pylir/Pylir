//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Bytecode/BytecodeWriter.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/DebugCounter.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/Timing.h>
#include <mlir/Support/ToolUtilities.h>
#include <mlir/Tools/ParseUtilities.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/ExternalModels/ExternalModels.hpp>
#include <pylir/Optimizer/Optimizer.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirMem/Transforms/Passes.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Passes.hpp>
#include <pylir/Optimizer/Transforms/Passes.hpp>

#include "Passes.hpp"
#include "TestDialect.hpp"

namespace
{
/// Perform the actions on the input file indicated by the command line flags
/// within the specified context.
///
/// This typically parses the main source file, runs zero or more optimization
/// passes, then prints the output.
///
mlir::LogicalResult performActions(llvm::raw_ostream& os, bool verifyPasses,
                                   const std::shared_ptr<llvm::SourceMgr>& sourceMgr, mlir::MLIRContext* context,
                                   mlir::PassPipelineFn passManagerSetupFn, bool emitBytecode)
{
    mlir::DefaultTimingManager tm;
    applyDefaultTimingManagerCLOptions(tm);
    mlir::TimingScope timing = tm.getRootScope();

    // Disable multi-threading when parsing the input file. This removes the
    // unnecessary/costly context synchronization when parsing.
    bool wasThreadingEnabled = context->isMultithreadingEnabled();
    context->disableMultithreading();

    // Prepare the parser config, and attach any useful/necessary resource
    // handlers. Unhandled external resources are treated as passthrough, i.e.
    // they are not processed and will be emitted directly to the output
    // untouched.
    mlir::PassReproducerOptions reproOptions;
    mlir::FallbackAsmResourceMap fallbackResourceMap;
    mlir::ParserConfig config(context, /*verifyAfterParse=*/false, &fallbackResourceMap);
    reproOptions.attachResourceParser(config);

    // Parse the input file and reset the context threading state.
    mlir::TimingScope parserTiming = timing.nest("Parser");
    mlir::OwningOpRef<mlir::Operation*> op = mlir::parseSourceFileForTool(sourceMgr, config, true);
    context->enableMultithreading(wasThreadingEnabled);
    if (!op)
    {
        return mlir::failure();
    }
    parserTiming.stop();

    // Prepare the pass manager, applying command-line and reproducer options.
    mlir::PassManager pm(context, op.get()->getName().getStringRef(), mlir::OpPassManager::Nesting::Implicit);
    pm.enableVerifier(verifyPasses);
    applyPassManagerCLOptions(pm);
    pm.enableTiming(timing);
    if (failed(reproOptions.apply(pm)) || failed(passManagerSetupFn(pm)))
    {
        return mlir::failure();
    }

    // Run the pipeline.
    if (failed(pm.run(*op)))
    {
        return mlir::failure();
    }

    // Print the output.
    mlir::TimingScope outputTiming = timing.nest("Output");
    if (emitBytecode)
    {
        mlir::BytecodeWriterConfig writerConfig(fallbackResourceMap);
        writeBytecodeToFile(op.get(), os, writerConfig);
    }
    else
    {
        mlir::AsmState asmState(op.get(), mlir::OpPrintingFlags(), /*locationMap=*/nullptr, &fallbackResourceMap);
        op.get()->print(os, asmState);
        os << '\n';
    }
    return mlir::success();
}

/// Parses the memory buffer.  If successfully, run a series of passes against
/// it and print the result.
mlir::LogicalResult processBuffer(llvm::raw_ostream& os, std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
                                  bool verifyDiagnostics, bool verifyPasses, bool allowUnregisteredDialects,
                                  bool emitBytecode, mlir::PassPipelineFn passManagerSetupFn,
                                  mlir::DialectRegistry& registry, llvm::ThreadPool* threadPool)
{
    // Tell sourceMgr about this buffer, which is what the parser will pick up.
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());

    // Create a context just for the current buffer. Disable threading on creation
    // since we'll inject the thread-pool separately.
    mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
    if (threadPool)
    {
        context.setThreadPool(*threadPool);
    }

    // Parse the input file.
    context.allowUnregisteredDialects(allowUnregisteredDialects);
    if (verifyDiagnostics)
    {
        context.printOpOnDiagnostic(false);
    }
    context.getDebugActionManager().registerActionHandler<mlir::DebugCounter>();

    // If we are in verify diagnostics mode then we have a lot of work to do,
    // otherwise just perform the actions without worrying about it.
    if (!verifyDiagnostics)
    {
        mlir::SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);
        return performActions(os, verifyPasses, sourceMgr, &context, passManagerSetupFn, emitBytecode);
    }

    mlir::SourceMgrDiagnosticVerifierHandler sourceMgrHandler(*sourceMgr, &context);

    // Do any processing requested by command line flags.  We don't care whether
    // these actions succeed or fail, we only care what diagnostics they produce
    // and whether they match our expectations.
    (void)performActions(os, verifyPasses, sourceMgr, &context, passManagerSetupFn, emitBytecode);

    // Verify the diagnostic handler to make sure that each of the diagnostics
    // matched.
    return sourceMgrHandler.verify();
}

mlir::LogicalResult mlirOptMain(llvm::raw_ostream& outputStream, std::unique_ptr<llvm::MemoryBuffer> buffer,
                                mlir::PassPipelineFn passManagerSetupFn, mlir::DialectRegistry& registry,
                                bool splitInputFile, bool verifyDiagnostics, bool verifyPasses,
                                bool allowUnregisteredDialects, bool emitBytecode)
{
    // The split-input-file mode is a very specific mode that slices the file
    // up into small pieces and checks each independently.
    // We use an explicit threadpool to avoid creating and joining/destroying
    // threads for each of the split.
    llvm::ThreadPool* threadPool = nullptr;

    // Create a temporary context for the sake of checking if
    // --mlir-disable-threading was passed on the command line.
    // We use the thread-pool this context is creating, and avoid
    // creating any thread when disabled.
    mlir::MLIRContext threadPoolCtx;
    if (threadPoolCtx.isMultithreadingEnabled())
    {
        threadPool = &threadPoolCtx.getThreadPool();
    }

    auto chunkFn = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer, llvm::raw_ostream& os)
    {
        return processBuffer(os, std::move(chunkBuffer), verifyDiagnostics, verifyPasses, allowUnregisteredDialects,
                             emitBytecode, passManagerSetupFn, registry, threadPool);
    };
    return mlir::splitAndProcessBuffer(std::move(buffer), chunkFn, outputStream, splitInputFile,
                                       /*insertMarkerInOutput=*/true);
}

} // namespace

int main(int argc, char** argv)
{
    mlir::registerAllPasses();

    mlir::DialectRegistry registry;
    registry.insert<pylir::Mem::PylirMemDialect, pylir::Py::PylirPyDialect, pylir::test::TestDialect>();
    mlir::registerAllDialects(registry);

    pylir::registerExternalModels(registry);

    pylir::registerConversionPasses();
    pylir::registerTransformPasses();
    pylir::Py::registerTransformPasses();
    pylir::Mem::registerTransformsPasses();
    pylir::test::registerTestPasses();
    pylir::registerOptimizationPipelines();

    // Reimplementation of MlirOptMain overload that sets up all the command line options. The purpose of doing so is
    // to implicitly run `pylir-finalize-ref-attrs` pass over the whole module before any other compiler passes are run.

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

    static llvm::cl::opt<bool> dumpPassPipeline{
        "dump-pass-pipeline", llvm::cl::desc("Print the pipeline that will be run"), llvm::cl::init(false)};

    llvm::InitLLVM y(argc, argv);

    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    mlir::registerDefaultTimingManagerCLOptions();
    mlir::DebugCounter::registerCLOptions();
    mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run", "p");

    // Build the list of dialects as a header for the --help message.
    std::string helpHeader = "pylir-opt\nAvailable Dialects: ";
    {
        llvm::raw_string_ostream os(helpHeader);
        interleaveComma(registry.getDialectNames(), os, [&](auto name) { os << name; });
    }

    // Form a new command line which injects the pylir-finalize-ref-attrs pass as first pass in the pipeline.
    // This is done by detecting use of -p and -pass-pipline and inserting it as very first pass there, or by simply
    // adding --pylir-finalize-ref-attrs as very first arg on the command line.
    std::vector<const char*> args(argv, argv + argc);
    std::string maybeNewPipeline;
    bool hadPipeline = false;
    for (auto iter = args.begin(); iter != args.end();)
    {
        std::string_view text = *iter;
        std::string_view pipeline;
        auto end = std::next(iter);
        if (text.substr(0, std::string_view("-p=").size()) == "-p=")
        {
            pipeline = text.substr(std::string_view("-p=").size());
        }
        else if (text.substr(0, std::string_view("-pass-pipeline=").size()) == "-pass-pipeline=")
        {
            pipeline = text.substr(std::string_view("-pass-pipeline=").size());
        }
        else if (text == "-p" || text == "-pass-pipeline")
        {
            if (std::next(iter) == args.end())
            {
                iter++;
                continue;
            }
            pipeline = *std::next(iter);
            end = std::next(end);
        }
        else
        {
            iter++;
            continue;
        }

        auto pos = pipeline.find('(');
        if (pos == std::string_view::npos)
        {
            iter++;
            continue;
        }
        maybeNewPipeline += "-pass-pipeline=";
        maybeNewPipeline += pipeline.substr(0, pos);
        maybeNewPipeline += '(';
        maybeNewPipeline += "pylir-finalize-ref-attrs, ";
        maybeNewPipeline += pipeline.substr(pos + 1);
        hadPipeline = true;
        iter = args.erase(iter, end);
        args.insert(iter, maybeNewPipeline.c_str());
        break;
    }
    if (!hadPipeline)
    {
        args.insert(std::next(args.begin()), "--pylir-finalize-ref-attrs");
    }

    // Parse pass names in main to ensure static initialization completed.
    llvm::cl::ParseCommandLineOptions(args.size(), args.data(), helpHeader);

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

    if (failed(::mlirOptMain(
            output->os(), std::move(file),
            [&](mlir::PassManager& pm)
            {
                if (mlir::failed(passPipeline.addToPipeline(pm,
                                                            [&](const llvm::Twine& msg)
                                                            {
                                                                mlir::emitError(mlir::UnknownLoc::get(pm.getContext()))
                                                                    << msg;
                                                                return mlir::failure();
                                                            })))
                {
                    return mlir::failure();
                }
                if (dumpPassPipeline)
                {
                    pm.dump();
                    llvm::errs() << "\n";
                }
                return mlir::success();
            },
            registry, splitInputFile, verifyDiagnostics, verifyPasses, allowUnregisteredDialects, emitBytecode)))
    {
        return -1;
    }

    // Keep the output file if the invocation of MlirOptMain was successful.
    output->keep();
}
