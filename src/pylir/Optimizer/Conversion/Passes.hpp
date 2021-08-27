#pragma once

#include <mlir/Pass/Pass.h>

#include "PylirMemToLLVMIR/PylirMemToLLVMIR.hpp"

namespace pylir
{
#define GEN_PASS_REGISTRATION
#include "pylir/Optimizer/Conversion/Passes.h.inc"

} // namespace pylir
