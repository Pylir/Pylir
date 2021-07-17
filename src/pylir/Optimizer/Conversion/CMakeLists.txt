
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(PylirConversionPassIncGen)

add_mlir_conversion_library(PylirToLLVM
        PylirToLLVM.cpp

        DEPENDS
        PylirConversionPassIncGen

        LINK_LIBS PUBLIC PylirDialect MLIRLLVMIR)
