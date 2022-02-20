
add_mlir_conversion_library(PylirToLLVMIR
        PylirToLLVMIR.cpp
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
        MLIRControlFlowToLLVM
        )

if (PYLIR_USE_PCH)
    target_link_libraries(PylirToLLVMIR PRIVATE COMMON_PCH)
    target_precompile_headers(PylirToLLVMIR PRIVATE <mlir/IR/Operation.h> <mlir/IR/BuiltinOps.h>)
endif ()
