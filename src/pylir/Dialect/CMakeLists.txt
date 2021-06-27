
add_mlir_dialect(PylirOps pylir)

add_mlir_dialect_library(PylirDialect
        PylirDialect.cpp
        PylirOps.cpp

        DEPENDS
        MLIRPylirOpsIncGen

        LINK_LIBS PUBLIC MLIRIR
        )
