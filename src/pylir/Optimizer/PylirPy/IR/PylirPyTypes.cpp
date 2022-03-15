#include "PylirPyTypes.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include "PylirPyDialect.hpp"

#define GET_TYPEDEF_CLASSES
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"

void pylir::Py::PylirPyDialect::initializeTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"
        >();
}
