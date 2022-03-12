
#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SubElementInterfaces.h>

#include <pylir/Optimizer/PylirPy/Util/Builtins.hpp>
#include <pylir/Support/BigInt.hpp>

#include <map>

#include "IntAttrInterface.hpp"
#include "ObjectAttrInterface.hpp"

#define GET_ATTRDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.h.inc"
