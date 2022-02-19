
add_library(PylirMarkAndSweep STATIC API.cpp MarkAndSweep.cpp SegregatedFreeList.cpp BestFitTree.cpp)
target_link_libraries(PylirMarkAndSweep PUBLIC PylirRuntime)
