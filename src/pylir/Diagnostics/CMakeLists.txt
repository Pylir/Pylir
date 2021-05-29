
add_library(Diagnostics DiagnosticsBuilder.cpp)
target_link_libraries(Diagnostics PRIVATE Support PUBLIC fmt)
