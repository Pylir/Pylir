
add_library(PylirLLVMPasses PlaceStatepoints.cpp PylirGC.cpp)
target_link_libraries(PylirLLVMPasses PUBLIC LLVMCore PRIVATE LLVMAnalysis LLVMCodeGen)
