# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

macro(pylir_obj_compile TARGET SOURCE)

    file(RELATIVE_PATH TargetRel "${CMAKE_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}")

    if (CMAKE_GENERATOR MATCHES "Ninja")
        # See https://cmake.org/cmake/help/latest/policy/CMP0116.html as to why this is currently relative.
        set(depfile_cmd -M ${TargetRel}.d)
        set(custom_command_extra DEPFILE ${TargetRel}.d)
    else ()
        get_target_property(depends_extra pylir-stdlib SOURCES)
        get_target_property(targetSourceDir pylir-stdlib SOURCE_DIR)
        list(TRANSFORM depends_extra PREPEND "${targetSourceDir}/")
    endif ()

    get_filename_component(SourceAbs ${SOURCE} REALPATH)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}
      COMMAND pylir ${SourceAbs} -fpie -c -o ${TargetRel} $<$<CONFIG:Release>:-O3> -I ${PROJECT_SOURCE_DIR}/src/python ${depfile_cmd}
      COMMENT "Building PY object ${TargetRel}"
      DEPENDS ${SourceAbs} pylir ${depends_extra}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      ${custom_command_extra}
      USES_TERMINAL
    )
    set_source_files_properties(${TARGET} PROPERTIES EXTERNAL_OBJECT true GENERATED true)
endmacro()
