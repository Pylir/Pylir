
set(LLVM_TARGET_DEFINITIONS PylirOps.td)
mlir_tablegen(PylirOps.h.inc -gen-op-decls)
mlir_tablegen(PylirOps.cpp.inc -gen-op-defs)
mlir_tablegen(PylirOpsTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PylirOpsTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(PylirOpsDialect.h.inc -gen-dialect-decls -dialect=pylir)
mlir_tablegen(PylirOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(PylirOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRPylirOpsIncGen)
add_dependencies(mlir-headers MLIRPylirOpsIncGen)

add_mlir_dialect_library(PylirDialect
        PylirDialect.cpp
        PylirOps.cpp
        PylirAttributes.cpp

        DEPENDS
        MLIRPylirOpsIncGen

        LINK_LIBS PUBLIC MLIRIR Support MLIRInferTypeOpInterface
        )
