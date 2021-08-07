
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(PylirConversionPassIncGen)


add_mlir_conversion_library(PylirToLLVM
        PylirToLLVM.cpp
        WinX64.cpp
        X86_64.cpp
        CABI.cpp

        DEPENDS
        PylirConversionPassIncGen

        LINK_LIBS PUBLIC PylirDialect MLIRLLVMIR MLIRPass MLIRTransforms MLIRStandardToLLVM)

if (PYLIR_USE_PCH)
    target_link_libraries(PylirToLLVM PRIVATE COMMON_PCH)
    target_precompile_headers(PylirToLLVM PRIVATE <mlir/IR/Operation.h> <mlir/IR/BuiltinOps.h>)
endif ()
