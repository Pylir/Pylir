
#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace pylir
{
std::unique_ptr<mlir::Pass> createLoadForwardingPass();

#define GEN_PASS_REGISTRATION
#include "pylir/Optimizer/Transforms/Passes.h.inc"

} // namespace pylir
