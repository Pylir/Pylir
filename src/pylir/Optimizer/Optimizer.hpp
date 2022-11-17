//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace pylir
{
/// Registers optimization pipelines used in pylir, making them available in the pass pipeline syntax.
/// There are currently three pipelines registered:
/// * "pylir-minimum", which does the minimum lowering of the output of pylir, to a state that can be converted to LLVM.
/// * "pylir-optimize", which is the full optimization pipeline used by pylir, also ending in a state that can be
///     lowered to LLVM.
/// * "pylir-llvm", which is capable of taking the lowered output of either of the two pipelines, and fully lower it to
///     the LLVM IR Dialect. This pipeline also has the following options:
///     - "target-triple": string that should be used as the LLVM target triple in the LLVM module.
///     - "data-layout": string that should be used as the LLVM data layout in the LLVM module.
///
void registerOptimizationPipelines();
} // namespace pylir