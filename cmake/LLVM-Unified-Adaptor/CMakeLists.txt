# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file is meant to be included as an external project when using a unified
# Pylir + LLVM build. Its purpose is to gather variable values that would
# normally be present in the LLVMConfig.cmake file of an installed LLVM and
# escape them to the caller code through the use of a global property.

set(MLIR_MAIN_SRC_DIR "${LLVM_EXTERNAL_MLIR_SOURCE_DIR}")
set(LLD_SOURCE_DIR "${LLVM_EXTERNAL_LLD_SOURCE_DIR}")
set(LLD_INCLUDE_DIRS "${LLD_SOURCE_DIR}/include;${LLVM_BINARY_DIR}/tools/lld/include")

# Variables whose name in LLVMConfig and value correspond.
set(forward_as_is
  LLD_INCLUDE_DIRS
  LLVM_ALL_TARGETS
  LLVM_BINARY_DIR
  LLVM_CMAKE_DIR
  LLVM_DEFINITIONS
  LLVM_ENABLE_RTTI
  LLVM_HOST_TRIPLE
  LLVM_NATIVE_ARCH
  LLVM_LIBRARY_DIR
  LLVM_TARGETS_TO_BUILD
  LLVM_TARGET_TRIPLE
  LLVM_TOOLS_BINARY_DIR
  MLIR_MAIN_SRC_DIR
)

foreach (var IN LISTS forward_as_is)
  # Make lists passable by replacing the semicolon with just a comma
  string(FIND "${${var}}" "," comma_index)
  if (NOT comma_index EQUAL -1)
    message(FATAL "Variable values with commas not supported")
  endif ()
  string(REPLACE ";" "," var_value "${${var}}")
  set_property(GLOBAL APPEND
    PROPERTY PYLIR_PROPAGATED_LLVM_FLAGS
    "${var}=${var_value}")
endforeach ()

set_property(GLOBAL APPEND
  PROPERTY PYLIR_PROPAGATED_LLVM_FLAGS
  "LLVM_INCLUDE_DIRS=${LLVM_MAIN_INCLUDE_DIR},${LLVM_INCLUDE_DIR}")

set_property(GLOBAL APPEND
  PROPERTY PYLIR_PROPAGATED_LLVM_FLAGS
  "MLIR_CMAKE_DIR=${MLIR_MAIN_SRC_DIR}/cmake/modules")

set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
set_property(GLOBAL APPEND
  PROPERTY PYLIR_PROPAGATED_LLVM_FLAGS
  "MLIR_INCLUDE_DIRS=${MLIR_MAIN_SRC_DIR}/include,${MLIR_GENERATED_INCLUDE_DIR}")
