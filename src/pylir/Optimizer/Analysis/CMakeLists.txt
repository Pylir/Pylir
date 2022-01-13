
add_library(PylirAnalysis MemorySSA.cpp)
target_link_libraries(PylirAnalysis PUBLIC MLIRPass MLIRAnalysis PRIVATE PylirCaptureInterface)
