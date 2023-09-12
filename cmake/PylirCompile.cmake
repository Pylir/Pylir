# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Use pylir to create an object file from any kind of source file supported by the compiler.
#
# pylir_obj_compile(
#   TARGET name
#       Filename of the object file that should be created
#   SOURCE file
#       Sourcefile that should be compiled
#   FLAGS flags...
#       Optional list of extra flags that should be passed to pylir
#   DEPENDS
#       Optional list of extra dependencies
macro(pylir_obj_compile)
  cmake_parse_arguments(ARG "" "TARGET;SOURCE" "FLAGS;DEPENDS" ${ARGN})
  
  file(RELATIVE_PATH TargetRel "${CMAKE_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}")
  
  if (CMAKE_GENERATOR MATCHES "Ninja|Makefiles")
    # See https://cmake.org/cmake/help/latest/policy/CMP0116.html as to why this is currently relative.
    set(depfile_cmd -M ${TargetRel}.d)
    set(custom_command_extra DEPFILE ${TargetRel}.d)
  else ()
    get_target_property(depends_extra pylir-stdlib SOURCES)
    get_target_property(targetSourceDir pylir-stdlib SOURCE_DIR)
    list(TRANSFORM depends_extra PREPEND "${targetSourceDir}/")
  endif ()
  
  get_filename_component(SourceExt ${ARG_SOURCE} EXT)
  if (${SourceExt} STREQUAL ".ll")
    set(LANG "LLVM")
  elseif (${SourceExt} STREQUAL ".mlir")
    set(LANG "MLIR")
  else ()
    set(LANG "PY")
  endif ()
  
  # TODO: Figure out a more cooperative and cmake informed way of doing this.
  set(pie_arg)
  if (NOT APPLE)
    set(pie_arg -fpie)
  endif ()
  
  set(sanitizer_arg)
  if (PYLIR_SANITIZERS)
    set(sanitizer_arg "-Xsanitize=${PYLIR_SANITIZERS}")
  endif ()
  
  get_filename_component(SourceAbs ${ARG_SOURCE} REALPATH)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}
    COMMAND pylir ${SourceAbs} ${pie_arg} ${sanitizer_arg} -c -o ${TargetRel} $<$<CONFIG:Release>:-O3>
    -I ${PROJECT_SOURCE_DIR}/src/python
    ${ARG_FLAGS} ${depfile_cmd}
    COMMENT "Building ${LANG} object ${TargetRel}"
    DEPENDS ${SourceAbs} pylir ${depends_extra} ${ARG_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    ${custom_command_extra}
  )
  set_source_files_properties(${ARG_TARGET} PROPERTIES EXTERNAL_OBJECT true GENERATED true)
endmacro()
