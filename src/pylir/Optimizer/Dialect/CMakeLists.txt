
set(LLVM_TARGET_DEFINITIONS PylirOps.td)
mlir_tablegen(PylirOps.h.inc -gen-op-decls)
mlir_tablegen(PylirOps.cpp.inc -gen-op-defs)
mlir_tablegen(PylirOpsTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PylirOpsTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(PylirOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(PylirOpsAttributes.cpp.inc -gen-attrdef-defs)
mlir_tablegen(PylirOpsDialect.h.inc -gen-dialect-decls -dialect=pylir)
mlir_tablegen(PylirOpsDialect.cpp.inc -gen-dialect-defs -dialect=pylir)
mlir_tablegen(PylirOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(PylirOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(PylirOpsIncGen)
add_dependencies(mlir-headers PylirOpsIncGen)

add_mlir_dialect_library(PylirDialect
        PylirDialect.cpp
        PylirOps.cpp
        PylirTypeObjects.cpp

        DEPENDS
        PylirOpsIncGen

        LINK_LIBS PUBLIC MLIRIR Support MLIRInferTypeOpInterface MLIRStandard MLIRMemRef
        )
