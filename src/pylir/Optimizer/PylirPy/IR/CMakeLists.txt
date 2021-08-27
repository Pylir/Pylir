
set(LLVM_TARGET_DEFINITIONS PylirPyOps.td)
mlir_tablegen(PylirPyOps.h.inc -gen-op-decls)
mlir_tablegen(PylirPyOps.cpp.inc -gen-op-defs)
mlir_tablegen(PylirPyOpsTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PylirPyOpsTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(PylirPyOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(PylirPyOpsAttributes.cpp.inc -gen-attrdef-defs)
mlir_tablegen(PylirPyOpsDialect.h.inc -gen-dialect-decls -dialect="pylir.py")
mlir_tablegen(PylirPyOpsDialect.cpp.inc -gen-dialect-defs -dialect="pylir.py")
mlir_tablegen(PylirPyOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(PylirPyOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(PylirPyOpsIncGen)
add_dependencies(mlir-headers PylirPyOpsIncGen)

add_mlir_dialect_library(PylirPyDialect
        PylirPyDialect.cpp
        PylirPyOps.cpp

        DEPENDS
        PylirPyOpsIncGen

        LINK_LIBS PUBLIC MLIRIR Support MLIRInferTypeOpInterface MLIRStandard
        )
if (PYLIR_USE_PCH)
    target_link_libraries(PylirPyDialect PRIVATE COMMON_PCH)
    target_precompile_headers(PylirPyDialect PRIVATE <mlir/IR/Operation.h> <mlir/IR/BuiltinOps.h>)
endif ()

