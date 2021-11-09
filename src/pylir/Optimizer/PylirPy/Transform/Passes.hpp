
#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace pylir::Py
{
std::unique_ptr<mlir::Pass> createExpandPyDialectPass();

#define GEN_PASS_REGISTRATION
#include "pylir/Optimizer/PylirPy/Transform/Passes.h.inc"

} // namespace pylir::Py
