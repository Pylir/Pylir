//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/Option/Arg.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ThreadPool.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Target/TargetMachine.h>

#include <pylir/Diagnostics/Document.hpp>
#include <pylir/Parser/Syntax.hpp>

#include <list>
#include <memory>
#include <mutex>
#include <optional>

#include "CommandLine.hpp"
#include "DiagnosticsVerifier.hpp"
#include "Toolchain.hpp"

namespace pylir {
struct CodeGenOptions;

class CompilerInvocation {
  std::unique_ptr<llvm::ThreadPoolInterface> m_threadPool;
  std::optional<mlir::MLIRContext> m_mlirContext;
  std::unique_ptr<llvm::LLVMContext> m_llvmContext;
  std::list<Diag::Document> m_documents;
  std::list<Syntax::FileInput> m_fileInputs;
  std::unique_ptr<llvm::TargetMachine> m_targetMachine;
  llvm::raw_pwrite_stream* m_output = nullptr;
  std::optional<llvm::sys::fs::TempFile> m_tempFile;
  std::optional<llvm::raw_fd_ostream> m_outFileStream;
  std::string m_compileStepOutputFilename;
  std::string m_actionOutputFilename;
  DiagnosticsVerifier* m_verifier;

  enum FileType { Python, MLIR, LLVM };

public:
  enum Action { SyntaxOnly, ObjectFile, Assembly, Link };

private:
  void ensureMLIRContext();

  mlir::LogicalResult ensureLLVMInit(const llvm::opt::InputArgList& args,
                                     const pylir::Toolchain& toolchain);

  mlir::LogicalResult ensureOutputStream(const llvm::opt::InputArgList& args,
                                         Action action,
                                         cli::CommandLine& commandLine);

  mlir::LogicalResult finalizeOutputStream(mlir::LogicalResult result,
                                           cli::CommandLine& commandLine);

  mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
  codegenPythonToMLIR(const llvm::opt::InputArgList& args,
                      const cli::CommandLine& commandLine,
                      Diag::DiagnosticsManager& diagManager,
                      Diag::DiagnosticsDocManager<>& mainModuleDiagManager);

  mlir::LogicalResult
  ensureTargetMachine(const llvm::opt::InputArgList& args,
                      cli::CommandLine& commandLine,
                      const pylir::Toolchain& toolchain,
                      std::optional<llvm::Triple> triple = {});

  mlir::LogicalResult compilation(llvm::opt::Arg* inputFile,
                                  cli::CommandLine& commandLine,
                                  const pylir::Toolchain& toolchain,
                                  CompilerInvocation::Action action,
                                  Diag::DiagnosticsManager& diagManager);

  Diag::Document& addDocument(std::string_view content, std::string filename);

public:
  explicit CompilerInvocation(DiagnosticsVerifier* verifier)
      : m_verifier(verifier) {}

  mlir::LogicalResult executeAction(llvm::opt::Arg* inputFile,
                                    cli::CommandLine& commandLine,
                                    const pylir::Toolchain& toolchain,
                                    CompilerInvocation::Action action,
                                    Diag::DiagnosticsManager& diagManager);
};

} // namespace pylir
