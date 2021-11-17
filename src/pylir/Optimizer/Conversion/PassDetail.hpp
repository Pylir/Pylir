
#pragma once

#include <mlir/Pass/Pass.h>

#include "Passes.hpp"

namespace pylir::Py
{
class PylirPyDialect;
}

namespace pylir::Mem
{
class PylirMemDialect;
}

namespace mlir::LLVM
{
class LLVMDialect;
}

namespace pylir
{
#define GEN_PASS_CLASSES
#include <pylir/Optimizer/Conversion/Passes.h.inc>
} // namespace pylir
