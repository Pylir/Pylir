
#include "TestDialect.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

void pylir::test::TestDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "TestDialectOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "TestDialectTypes.cpp.inc"
        >();
    addAttributes<
#define GET_ATTRDEF_LIST
#include "TestDialectAttributes.cpp.inc"
        >();
}

#include "TestDialect.cpp.inc"

#define GET_OP_CLASSES
#include "TestDialectOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TestDialectTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "TestDialectAttributes.cpp.inc"
