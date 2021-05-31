
include(CTest)
enable_testing()

if (MSVC AND ("${PYLIR_SANITIZER}" MATCHES ".*address.*"))
    add_compile_definitions(CATCH_CONFIG_NO_WINDOWS_SEH)
endif ()

add_subdirectory(Diagnostics)
add_subdirectory(Lexer)
