
set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(PylirMainOptsTableGen)

llvm_map_components_to_libnames(llvm_all ${LLVM_TARGETS_TO_BUILD} Passes)
llvm_map_components_to_libnames(llvm_options Option)

add_library(PylirMain PylirMain.cpp)
target_link_libraries(PylirMain PUBLIC Diagnostics ${llvm_options} PRIVATE CodeGen PylirMemToLLVMIR MLIRToLLVMIRTranslationRegistration ${llvm_all} PylirPyTransforms PylirPyToPylirMem PylirMemToLLVMIR)
add_dependencies(PylirMain PylirMainOptsTableGen)
target_compile_definitions(PylirMain PRIVATE PYLIR_VERSION="${PROJECT_VERSION}")
