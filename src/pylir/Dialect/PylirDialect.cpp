#include "PylirDialect.hpp"

#include "PylirOps.hpp"

void pylir::Dialect::PylirDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "pylir/Dialect/PylirOps.cpp.inc"
        >();
}
