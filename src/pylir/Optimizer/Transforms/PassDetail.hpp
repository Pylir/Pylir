#pragma once

#include <mlir/Pass/Pass.h>

#include <pylir/Optimizer/Analysis/MemorySSAIR.hpp>

namespace pylir
{
#define GEN_PASS_CLASSES
#include <pylir/Optimizer/Transforms/Passes.h.inc>
} // namespace pylir
