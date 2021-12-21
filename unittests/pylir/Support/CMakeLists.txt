
include(Catch)

add_executable(support_tests main.cpp bigint_tests.cpp text_tests.cpp)
target_link_libraries(support_tests PylirSupport)
catch_discover_tests(support_tests)
