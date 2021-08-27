#pragma once

#include <mlir/Pass/Pass.h>

#include "PylirMemToLLVMIR/PylirMemToLLVMIR.hpp"

namespace pylir::Dialect
{
#define GEN_PASS_REGISTRATION
#include "pylir/Optimizer/Conversion/Passes.h.inc"

} // namespace pylir::Dialect
