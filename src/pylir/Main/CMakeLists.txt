
llvm_map_components_to_libnames(llvm_all ${LLVM_TARGETS_TO_BUILD})

add_library(PylirMain PylirMain.cpp)
target_link_libraries(PylirMain PRIVATE CodeGen PylirToLLVM MLIRToLLVMIRTranslationRegistration ${llvm_all})
