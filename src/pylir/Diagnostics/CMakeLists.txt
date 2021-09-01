
add_library(Diagnostics DiagnosticsBuilder.cpp Document.cpp)
target_link_libraries(Diagnostics PRIVATE PylirSupport PUBLIC fmt)
if (PYLIR_USE_PCH)
    target_link_libraries(Diagnostics PRIVATE COMMON_PCH)
    target_precompile_headers(Diagnostics PRIVATE <fmt/format.h>)
endif ()
