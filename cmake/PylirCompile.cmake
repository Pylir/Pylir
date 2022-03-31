
macro(pylir_obj_compile TARGET SOURCE)
    get_filename_component(SourceAbs ${SOURCE} REALPATH)
    file(RELATIVE_PATH TargetRel "${CMAKE_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}")
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}
            COMMAND pylir ${SourceAbs} -fpie -c -o ${CMAKE_CURRENT_BINARY_DIR}/${TARGET} $<$<CONFIG:Release>:-O3>
            COMMENT "Building PY object ${TargetRel}"
            DEPENDS ${SourceAbs} pylir
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_source_files_properties(${TARGET} PROPERTIES EXTERNAL_OBJECT true GENERATED true)
endmacro()
