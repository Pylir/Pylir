#  Licensed under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Create phony target used to collect build dependencies.
add_custom_target(pylir-doc)

set(PYLIR_PREPROCESS_MLIR_MD "${CMAKE_CURRENT_SOURCE_DIR}/preprocess_mlir_md.py"
  PARENT_SCOPE)

# Delete the source directory as there may be leftover files from previous
# copies. Copy over the files in this directory and the preprocessed TableGen
# output afterwards.
add_custom_target(create-build-dir
  "${CMAKE_COMMAND}" -E rm -rf "${CMAKE_CURRENT_BINARY_DIR}/source"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
  "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/source"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
  "${CMAKE_CURRENT_BINARY_DIR}/TableGen"
  "${CMAKE_CURRENT_BINARY_DIR}/source/TableGen"
  COMMENT "Copying docs sources")
set_target_properties(create-build-dir
  PROPERTIES ADDITIONAL_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/source")

add_dependencies(create-build-dir pylir-doc)

add_custom_target(docs
  ${Python3_EXECUTABLE} -m sphinx.cmd.build -b html . ../build -W --keep-going -n
  COMMENT "Building Sphinx docs"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/source)
set_target_properties(docs
  PROPERTIES ADDITIONAL_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/build")
add_dependencies(docs create-build-dir)
