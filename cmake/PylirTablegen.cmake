
function(pylir_tablegen ofn)
  tablegen(PYLIR ${ARGV})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    PARENT_SCOPE)
  
  # Get the current set of include paths for this td file.
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  # Filter out any empty include items.
  list(REMOVE_ITEM tblgen_includes "")
  
  # Build the absolute path for the current input file.
  if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else ()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif ()
  
  # Append the includes used for this file to the tablegen_compile_commands
  # file.
  file(APPEND ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml
    "--- !FileInfo:\n"
    "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
    "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n"
  )
endfunction()
