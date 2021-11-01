
add_library(PylirPyUtil Util.cpp)
target_link_libraries(PylirPyUtil PUBLIC PylirPyDialect PRIVATE MLIRStandard)
if (PYLIR_USE_PCH)
    target_link_libraries(PylirPyUtil PRIVATE COMMON_PCH)
    target_precompile_headers(PylirPyUtil PRIVATE <mlir/IR/Operation.h> <mlir/IR/BuiltinOps.h>)
endif ()
