//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <pylir/Diagnostics/DiagnosticsManager.hpp>
#include <pylir/Diagnostics/LocationProvider.hpp>
#include <pylir/Parser/Syntax.hpp>

#include <cstddef>
#include <functional>
#include <string>
#include <utility>

namespace pylir {

/// Option struct used to customize the code generation behaviour.
struct CodeGenOptions {
  /// Callback used to tell the caller that a Python module is being used and
  /// should be compiled. 'absoluteModule' contains the Python qualifier for
  /// the module in 'qualifier {`.` qualifier }' form. 'diagnostics' and
  /// 'location' should be used to emit any diagnostic while trying to find the
  /// python module.
  ///
  /// TODO: This should be in the parser/semantic analysis.
  std::function<void(llvm::StringRef absoluteModule,
                     Diag::DiagnosticsDocManager<>* diagnostics,
                     Diag::LazyLocation location)>
      moduleLoadCallback;

  /// Python qualifier of the module being compiled. This must be '__main__' for
  /// the main module.
  std::string qualifier;

  /// Whether '__main__' should import 'builtins'.
  bool implicitBuiltinsImport = true;
};

/// Performs code generation from the AST of a python module, to MLIR.
/// The returned MLIR contains a mix of the 'pyHIR', 'py', 'cf' and 'arith'
/// dialect, with the large majority of the code being nested within a
/// 'pyHIR.init' op. The module itself does not verify until linked with all
/// its imported python modules into one MLIR module.
///
/// 'options' is used to customize the compilation process, most importantly
/// providing a callback to request module loading and more.
/// 'docManager' is used to emit any diagnostics during the compilation process.
/// The contents of the MLIR module are unspecified if any diagnostics were
/// emitted.
///
/// TODO: CodeGen should not emit diagnostics.
mlir::OwningOpRef<mlir::ModuleOp>
codegenModule(mlir::MLIRContext* context, const Syntax::FileInput& input,
              Diag::DiagnosticsDocManager<>& docManager,
              const CodeGenOptions& options);
} // namespace pylir
