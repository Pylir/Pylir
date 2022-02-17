
set(PYLIR_RUNTIME_BUILTINS "${CMAKE_CURRENT_BINARY_DIR}/builtins.o")

add_custom_command(
        OUTPUT ${PYLIR_RUNTIME_BUILTINS}
        COMMAND pylir ${CMAKE_CURRENT_SOURCE_DIR}/empty.py -fpie -c -o ${CMAKE_CURRENT_BINARY_DIR}/builtins.o
        COMMENT "Generating builtins for test"
        DEPENDS empty.py pylir
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set_source_files_properties(${PYLIR_RUNTIME_BUILTINS} PROPERTIES EXTERNAL_OBJECT true GENERATED true)
add_library(PylirTestRuntime STATIC ${PYLIR_RUNTIME_BUILTINS})
target_link_libraries(PylirTestRuntime PUBLIC PylirRuntime)

add_subdirectory(MarkAndSweep)
