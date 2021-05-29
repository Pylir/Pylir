
add_library(Diagnostics Diagnostics.cpp)
target_link_libraries(Diagnostics PRIVATE Support PUBLIC fmt)
