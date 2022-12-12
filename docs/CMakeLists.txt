
find_program(SPHINX_EXECUTABLE NAMES sphinx-build REQUIRED)

set(PYLIR_PREPROCESS_MLIR_MD "${CMAKE_CURRENT_SOURCE_DIR}/preprocess_mlir_md.py" PARENT_SCOPE)

add_custom_target(create-build-dir
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Copying docs sources")

add_custom_target(docs
        ${SPHINX_EXECUTABLE} -b html . build
        COMMENT "Building Sphinx docs"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(docs PROPERTIES ADDITIONAL_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/build")
add_dependencies(docs create-build-dir mlir-doc)
