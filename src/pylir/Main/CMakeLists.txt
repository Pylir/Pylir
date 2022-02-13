
set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(PylirMainOptsTableGen)

llvm_map_components_to_libnames(llvm_all ${LLVM_TARGETS_TO_BUILD} Passes)
llvm_map_components_to_libnames(llvm_options Option)

add_library(PylirMain PylirMain.cpp CommandLine.cpp CommandLine.hpp Toolchain.cpp Toolchain.hpp MinGWToolchain.cpp MinGWToolchain.hpp MSVCToolchain.cpp MSVCToolchain.hpp LinuxToolchain.cpp LinuxToolchain.hpp Version.cpp Version.hpp Distro.cpp Distro.hpp CompilerInvocation.cpp CompilerInvocation.hpp)
target_link_libraries(PylirMain PUBLIC Diagnostics ${llvm_options} PRIVATE CodeGen PylirMemToLLVMIR MLIRToLLVMIRTranslationRegistration ${llvm_all} PylirPyTransforms PylirPyToPylirMem PylirMemToLLVMIR PylirTransforms PylirLLVMPasses MLIRReconcileUnrealizedCasts MLIRArithmeticTransforms)
add_dependencies(PylirMain PylirMainOptsTableGen)
target_compile_definitions(PylirMain PRIVATE PYLIR_VERSION="${PROJECT_VERSION}" PRIVATE PYLIR_DEFAULT_SYSROOT="${PYLIR_DEFAULT_SYSROOT}")
if (PYLIR_EMBED_LLD)
    target_compile_definitions(PylirMain PRIVATE PYLIR_EMBEDDED_LLD)
    target_link_libraries(PylirMain PRIVATE lldCOFF lldELF lldWasm lldMachO lldMinGW)
    target_include_directories(PylirMain PRIVATE ${LLD_INCLUDE_DIRS})
endif ()
