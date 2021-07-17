
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(PylirConversionPassIncGen)

set(LLVM_TARGET_DEFINITIONS PylirToLLVMPatterns.td)
mlir_tablegen(PylirToLLVMPatterns.h.inc -gen-rewriters)
add_public_tablegen_target(PylirToLLVMPatternsIncGen)

add_mlir_conversion_library(PylirToLLVM
        PylirToLLVM.cpp

        DEPENDS
        PylirConversionPassIncGen
        PylirToLLVMPatternsIncGen

        LINK_LIBS PUBLIC PylirDialect MLIRLLVMIR)
