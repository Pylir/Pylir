//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Pass/PassOptions.h>

namespace pylir {

/// Pass options for the 'pylir-llvm' pass-pipeline.
struct PylirLLVMOptions : public mlir::PassPipelineOptions<PylirLLVMOptions> {
  Option<std::string> targetTriple{*this, "target-triple",
                                   llvm::cl::desc("LLVM target triple"),
                                   llvm::cl::init(LLVM_DEFAULT_TARGET_TRIPLE)};
  Option<std::string> dataLayout{*this, "data-layout",
                                 llvm::cl::desc("LLVM data layout"),
                                 llvm::cl::init("")};

  Option<bool> debugInfo{*this, "debug-info",
                         llvm::cl::desc("Whether to produce debug info"),
                         llvm::cl::init(false)};

  PylirLLVMOptions() = default;

  PylirLLVMOptions(llvm::StringRef targetTriple, llvm::StringRef dataLayout,
                   bool produceDebugInfo) {
    this->targetTriple = targetTriple.str();
    this->dataLayout = dataLayout.str();
    this->debugInfo = produceDebugInfo;
  }

  /// Prints the option struct options in a format suitable for directly
  /// appending to the pass pipeline name. In other words, this already includes
  /// the surrounding '{}'.
  std::string rendered() {
    std::string rendered;
    llvm::raw_string_ostream ss(rendered);
    print(ss);
    return rendered;
  }
};

/// Registers optimization pipelines used in pylir, making them available in the
/// pass pipeline syntax. There are currently three pipelines registered:
/// * "pylir-minimum", which does the minimum lowering of the output of pylir,
/// to a state that can be converted to LLVM.
/// * "pylir-optimize", which is the full optimization pipeline used by pylir,
/// also ending in a state that can be
///     lowered to LLVM.
/// * "pylir-llvm", which is capable of taking the lowered output of either of
/// the two pipelines, and fully lower it to
///     the LLVM IR Dialect. This pipeline also has the following options:
///     - "target-triple": string that should be used as the LLVM target triple
///     in the LLVM module.
///     - "data-layout": string that should be used as the LLVM data layout in
///     the LLVM module.
///
void registerOptimizationPipelines();
} // namespace pylir
