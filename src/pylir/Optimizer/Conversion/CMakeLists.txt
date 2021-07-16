
add_mlir_conversion_library(PylirToLLVM
        PylirToLLVM.cpp

        DEPENDS
        MLIRPylirOpsIncGen

        LINK_LIBS PUBLIC PylirDialect MLIRLLVMIR)
