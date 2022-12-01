
find_program(HUGO_EXECUTABLE NAMES hugo REQUIRED)

set(PYLIR_PREPROCESS_MLIR_MD "${CMAKE_CURRENT_SOURCE_DIR}/preprocess_mlir_md.py" PARENT_SCOPE)

add_custom_target(create-build-dir
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Copying docs sources")

add_custom_target(docs ALL
        ${HUGO_EXECUTABLE}
        -d "${PYLIR_BINARY_DIR}/docs/build"
        --quiet
        COMMENT "Building Hugo docs"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(docs create-build-dir mlir-doc)

add_custom_target(hugo-server ${HUGO_EXECUTABLE} server WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} USES_TERMINAL)
add_dependencies(hugo-server create-build-dir mlir-doc)