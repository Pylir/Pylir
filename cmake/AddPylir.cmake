
function(add_pylir_dialect dialect dialect_namespace)
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
  mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls --typedefs-dialect="${dialect_namespace}")
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs --typedefs-dialect="${dialect_namespace}")
  mlir_tablegen(${dialect}Attributes.h.inc -gen-attrdef-decls --attrdefs-dialect="${dialect_namespace}")
  mlir_tablegen(${dialect}Attributes.cpp.inc -gen-attrdef-defs --attrdefs-dialect="${dialect_namespace}")
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect="${dialect_namespace}")
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect="${dialect_namespace}")
  mlir_tablegen(${dialect}Enums.h.inc -gen-enum-decls)
  mlir_tablegen(${dialect}Enums.cpp.inc -gen-enum-defs)
  add_public_tablegen_target(${dialect}IncGen)
endfunction()

function(add_pylir_interface kind interface)
  cmake_parse_arguments(ARG "LIBRARY" "LIB_PREFIX" "LIB_DEPS" ${ARGN})
  set(lib_prefix)
  if (ARG_LIB_PREFIX)
    set(lib_prefix ${ARG_LIB_PREFIX})
  endif ()
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  string(TOLOWER ${kind} kind)
  mlir_tablegen(${interface}.h.inc -gen-${kind}-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-${kind}-interface-defs)
  add_public_tablegen_target(${lib_prefix}${interface}IncGen)

  if (NOT ARG_LIBRARY)
    return()
  endif ()

  set(link_deps MLIRIR)
  list(APPEND link_deps ${ARG_LIB_DEPS})

  add_library(${lib_prefix}${interface} ${interface}.cpp)
  add_dependencies(${lib_prefix}${interface} ${lib_prefix}${interface}IncGen)
  target_link_libraries(${lib_prefix}${interface} PUBLIC ${link_deps})
endfunction()

function(add_pylir_rewriter patterns)
  set(LLVM_TARGET_DEFINITIONS ${patterns}.td)
  mlir_tablegen(${patterns}.cpp.inc -gen-rewriters)
  add_public_tablegen_target(${patterns}IncGen)
endfunction()

function(add_pylir_passes file name)
  cmake_parse_arguments(ARG "" "PREFIX" "" ${ARGN})

  set(prefix)
  if (ARG_PREFIX)
    set(prefix ${ARG_PREFIX})
  endif ()

  set(LLVM_TARGET_DEFINITIONS ${file}.td)
  mlir_tablegen(${file}.h.inc -gen-pass-decls -name ${name})
  add_public_tablegen_target(${prefix}${name}PassIncGen)
endfunction()
