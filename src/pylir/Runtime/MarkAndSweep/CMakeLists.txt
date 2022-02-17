
add_library(PylirMarkAndSweep STATIC API.cpp MarkAndSweep.cpp SegregatedFreeList.cpp BestFitTree.cpp BestFitTree.hpp)
target_link_libraries(PylirMarkAndSweep PUBLIC PylirRuntime)
