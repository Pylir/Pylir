// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/Option/Arg.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Target/TargetMachine.h>

#include <pylir/Diagnostics/Document.hpp>
#include <pylir/Parser/Syntax.hpp>

#include <memory>
#include <list>
#include <mutex>
#include <optional>

#include "CommandLine.hpp"
#include "Toolchain.hpp"

namespace pylir
{
struct CodeGenOptions;

class CompilerInvocation
{
    std::optional<mlir::MLIRContext> m_mlirContext;
    std::unique_ptr<llvm::LLVMContext> m_llvmContext;
    std::list<Diag::Document> m_documents;
    std::list<Syntax::FileInput> m_fileInputs;
    std::unique_ptr<llvm::TargetMachine> m_targetMachine;
    llvm::raw_pwrite_stream* m_output = nullptr;
    std::optional<llvm::sys::fs::TempFile> m_outputFile;
    std::optional<llvm::raw_fd_ostream> m_outFileStream;
    std::string m_compileStepOutputFilename;
    std::string m_actionOutputFilename;

    enum FileType
    {
        Python,
        MLIR,
        LLVM
    };

public:
    enum Action
    {
        SyntaxOnly,
        ObjectFile,
        Assembly,
        Link
    };

private:
    void ensureMLIRContext(const llvm::opt::InputArgList& args);

    mlir::LogicalResult ensureLLVMInit(const llvm::opt::InputArgList& args, const pylir::Toolchain& toolchain);

    mlir::LogicalResult ensureOutputStream(const llvm::opt::InputArgList& args, Action action);

    mlir::LogicalResult finalizeOutputStream(mlir::LogicalResult result);

    void addOptimizationPasses(llvm::StringRef level, mlir::OpPassManager& manager);

    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> codegenPythonToMLIR(const llvm::opt::InputArgList& args,
                                                                           const cli::CommandLine& commandLine);

    mlir::LogicalResult ensureTargetMachine(const llvm::opt::InputArgList& args, const cli::CommandLine& commandLine,
                                            const pylir::Toolchain& toolchain,
                                            llvm::Optional<llvm::Triple> triple = {});

    mlir::LogicalResult compilation(llvm::opt::Arg* inputFile, const cli::CommandLine& commandLine,
                                    const pylir::Toolchain& toolchain, Action action);

public:
    CompilerInvocation() = default;

    mlir::LogicalResult executeAction(llvm::opt::Arg* inputFile, const cli::CommandLine& commandLine,
                                      const pylir::Toolchain& toolchain, Action action);
};

} // namespace pylir
