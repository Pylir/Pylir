
#include "PylirPyDialect.hpp"

#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsDialect.cpp.inc"

void pylir::Py::PylirPyDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOps.cpp.inc"
        >();
    addTypes<
#define GET_OP_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsTypes.cpp.inc"
        >();
    addAttributes<
#define GET_OP_LIST
#include "pylir/Optimizer/PylirPy/IR/PylirPyOpsAttributes.cpp.inc"
        >();
}
