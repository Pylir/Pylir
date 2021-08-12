
llvm_map_components_to_libnames(llvm_all ${LLVM_TARGETS_TO_BUILD} Passes)

add_executable(codegen_test_runner test_runner.cpp)
target_link_libraries(codegen_test_runner CodeGen PylirToLLVM MLIRToLLVMIRTranslationRegistration ${llvm_all})
