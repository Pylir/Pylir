
add_library(Diagnostics DiagnosticsBuilder.cpp Document.cpp Document.hpp)
target_link_libraries(Diagnostics PRIVATE Support PUBLIC fmt)
