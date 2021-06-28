#include "PylirDialect.hpp"

#include "PylirOps.hpp"

void pylir::Dialect::PylirDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "pylir/Dialect/PylirOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "pylir/Dialect/PylirOpsTypes.h.inc"
        >();
}
