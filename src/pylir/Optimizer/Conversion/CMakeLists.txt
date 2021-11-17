
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(PylirConversionPassIncGen)

add_subdirectory(PylirMemToLLVMIR)
add_subdirectory(PylirPyToPylirMem)
