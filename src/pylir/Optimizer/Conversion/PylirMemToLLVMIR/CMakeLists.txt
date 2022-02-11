
add_mlir_conversion_library(PylirMemToLLVMIR
        PylirMemToLLVMIR.cpp
        WinX64.cpp
        X86_64.cpp
        PlatformABI.cpp

        DEPENDS
        PylirConversionPassIncGen

        LINK_LIBS PUBLIC
        PylirMemDialect
        MLIRLLVMIR
        MLIRPass
        MLIRTransforms
        MLIRStandardToLLVM
        MLIRArithmeticToLLVM
        MLIRControlFlowToLLVM
        )

if (PYLIR_USE_PCH)
    target_link_libraries(PylirMemToLLVMIR PRIVATE COMMON_PCH)
    target_precompile_headers(PylirMemToLLVMIR PRIVATE <mlir/IR/Operation.h> <mlir/IR/BuiltinOps.h>)
endif ()
