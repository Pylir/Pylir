
include(PylirTablegen)

#
function(add_pylir_doc td_filename output_file output_directory command)
  if (NOT PYLIR_BUILD_DOCS)
    return()
  endif ()
  
  cmake_parse_arguments(ARG "" "" "PREPROCESS_ARGS" ${ARGN})
  
  set(LLVM_TARGET_DEFINITIONS ${td_filename})
  mlir_tablegen(${output_file}.md ${command} ${ARG_UNPARSED_ARGUMENTS})
  set(GEN_DOC_FILE
    "${PYLIR_BINARY_DIR}/docs/TableGen/${output_directory}${output_file}.md")
  add_custom_command(
    OUTPUT ${GEN_DOC_FILE}
    COMMAND ${Python3_EXECUTABLE} ${PYLIR_PREPROCESS_MLIR_MD}
    ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
    ${GEN_DOC_FILE}
    ${ARG_PREPROCESS_ARGS}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
    ${PYLIR_PREPROCESS_MLIR_MD})
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(pylir-doc ${output_file}DocGen)
endfunction()

#
function(add_pylir_dialect dialect dialect_namespace)
  
  cmake_parse_arguments(ARG "NO_DOC" "" "" ${ARGN})
  
  set(td_file)
  if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/${dialect}.td)
    set(td_file ${dialect}.td)
  elseif (EXISTS ${CMAKE_CURRENT_LIST_DIR}/${dialect}Ops.td)
    set(td_file ${dialect}Ops.td)
  elseif (EXISTS ${CMAKE_CURRENT_LIST_DIR}/${dialect}Dialect.td)
    set(td_file ${dialect}Dialect.td)
  else ()
    message(FATAL_ERROR "No *.td file found for ${dialect}")
  endif ()
  
  set(LLVM_TARGET_DEFINITIONS ${td_file})
  mlir_tablegen(${dialect}Ops.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}Ops.cpp.inc -gen-op-defs)
  pylir_tablegen(${dialect}OpsExtra.cpp.inc -gen-op-variable-decorators)
  mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls
    --typedefs-dialect="${dialect_namespace}")
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs
    --typedefs-dialect="${dialect_namespace}")
  mlir_tablegen(${dialect}Attributes.h.inc -gen-attrdef-decls
    --attrdefs-dialect="${dialect_namespace}")
  mlir_tablegen(${dialect}Attributes.cpp.inc -gen-attrdef-defs
    --attrdefs-dialect="${dialect_namespace}")
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls
    -dialect="${dialect_namespace}")
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs
    -dialect="${dialect_namespace}")
  if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/${dialect}Enums.td)
    set(LLVM_TARGET_DEFINITIONS ${dialect}Enums.td)
    mlir_tablegen(${dialect}Enums.h.inc -gen-enum-decls)
    mlir_tablegen(${dialect}Enums.cpp.inc -gen-enum-defs)
  endif ()
  add_public_tablegen_target(${dialect}IncGen)
  
  if (NOT ARG_NO_DOC)
    get_filename_component(output_file ${td_file} NAME_WE)
    add_pylir_doc(${td_file} ${dialect}Dialect Dialect/
      -gen-dialect-doc -dialect="${dialect_namespace}")
  endif ()
endfunction()

#
function(add_pylir_interface kind interface)
  cmake_parse_arguments(ARG "LIBRARY" "LIB_PREFIX;FILE" "LIB_DEPS" ${ARGN})
  set(lib_prefix)
  if (ARG_LIB_PREFIX)
    set(lib_prefix ${ARG_LIB_PREFIX})
  endif ()
  
  set(file ${interface})
  if (ARG_FILE)
    set(file ${ARG_FILE})
  endif ()
  
  set(LLVM_TARGET_DEFINITIONS ${file}.td)
  string(TOLOWER ${kind} kind)
  mlir_tablegen(${interface}.h.inc -gen-${kind}-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-${kind}-interface-defs)
  add_public_tablegen_target("${lib_prefix}${interface}IncGen")
  
  set(output_dir "${CMAKE_CURRENT_SOURCE_DIR}")
  cmake_path(RELATIVE_PATH output_dir BASE_DIRECTORY ${PYLIR_SOURCE_DIR}/src)
  
  add_pylir_doc(${file}.td "${file}${kind}" ${output_dir}/
    -gen-${kind}-interface-docs PREPROCESS_ARGS -strip-title -title-indent=1)
  
  if (NOT ARG_LIBRARY)
    return()
  endif ()
  
  set(link_deps MLIRIR)
  list(APPEND link_deps ${ARG_LIB_DEPS})
  
  add_library("${lib_prefix}${interface}" ${file}.cpp)
  add_dependencies("${lib_prefix}${interface}"
    "${lib_prefix}${interface}IncGen")
  target_link_libraries("${lib_prefix}${interface}" PUBLIC ${link_deps})
endfunction()

function(add_pylir_rewriter patterns)
  set(LLVM_TARGET_DEFINITIONS ${patterns}.td)
  mlir_tablegen(${patterns}.cpp.inc -gen-rewriters)
  add_public_tablegen_target(${patterns}IncGen)
endfunction()

#
function(add_pylir_passes file name)
  cmake_parse_arguments(ARG "NO_DOC" "PREFIX" "" ${ARGN})
  
  set(prefix)
  if (ARG_PREFIX)
    set(prefix ${ARG_PREFIX})
  endif ()
  
  set(LLVM_TARGET_DEFINITIONS ${file}.td)
  mlir_tablegen(${file}.h.inc -gen-pass-decls -name ${name})
  add_public_tablegen_target("${prefix}${name}PassIncGen")
  
  if (NOT ARG_NO_DOC)
    add_pylir_doc(${file}.td "${prefix}${name}" Passes/ -gen-pass-doc)
  endif ()
endfunction()
