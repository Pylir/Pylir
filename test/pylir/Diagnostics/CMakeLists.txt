
include(Catch)

add_executable(diagnostics_tests main.cpp diagnostics_tests.cpp document_tests.cpp)
target_link_libraries(diagnostics_tests Diagnostics)
catch_discover_tests(diagnostics_tests)
