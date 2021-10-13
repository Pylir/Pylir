
#pragma once

#include "ExpandPyDialect/ExpandPyDialect.hpp"

namespace pylir::Py
{
#define GEN_PASS_REGISTRATION
#include "pylir/Optimizer/PylirPy/Transform/Passes.h.inc"

} // namespace pylir::Py
