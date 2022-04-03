#pragma once

#include <mlir/Interfaces/CallInterfaces.h>

namespace pylir::Py
{
mlir::LogicalResult inlineCall(mlir::CallOpInterface call, mlir::CallableOpInterface callable);
}
