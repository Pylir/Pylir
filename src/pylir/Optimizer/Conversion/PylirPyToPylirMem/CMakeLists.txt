
set(LLVM_TARGET_DEFINITIONS PylirPyToPylirMem.td)
mlir_tablegen(PylirPyToPylirMem.cpp.inc -gen-rewriters)
add_public_tablegen_target(PylirPyToPylirMemIncGen)

add_mlir_conversion_library(PylirPyToPylirMem
        PylirPyToPylirMem.cpp

        DEPENDS
        PylirConversionPassIncGen
        PylirPyToPylirMemIncGen

        LINK_LIBS PUBLIC PylirPyDialect PylirMemDialect MLIRPass MLIRTransforms)

if (PYLIR_USE_PCH)
    target_link_libraries(PylirPyToPylirMem PRIVATE COMMON_PCH)
    target_precompile_headers(PylirPyToPylirMem PRIVATE <mlir/IR/Operation.h> <mlir/IR/BuiltinOps.h>)
endif ()
