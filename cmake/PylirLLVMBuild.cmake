# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Uses CPM to download and configure LLVM at the given revision.
function(add_required_llvm_build llvm_revision)
  include(CPM)

  if (CMAKE_VERSION VERSION_LESS "3.21")
    message(FATAL_ERROR "Building LLVM as part of Pylir requires at least version 3.21")
  endif ()

  # Set the default value of the MSVC runtime library to the same as CMake
  # as is documented here: https://cmake.org/cmake/help/latest/variable/CMAKE_MSVC_RUNTIME_LIBRARY.html
  # This is required to make the LLVM build use the same runtime library
  # regardless of the value of `PYLIR_LLVM_CMAKE_BUILD_TYPE`.
  set(msvc_runtime_default "MultiThreadedDLL")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(msvc_runtime_default "MultiThreadedDebugDLL")
  endif ()

  set(CMAKE_MSVC_RUNTIME_LIBRARY "${msvc_runtime_default}"
    CACHE STRING "MSVC Runtime library to use")

  # Purposefully enable console output as LLVM takes a long time to clone and
  # users would not get feedback otherwise.
  set(FETCHCONTENT_QUIET FALSE)
  # These variables have to be unset if updating the LLVM version or similar
  # as they may still point to the previous directory otherwise.
  unset(LLVM_EXTERNAL_LLD_SOURCE_DIR CACHE)
  unset(LLVM_EXTERNAL_MLIR_SOURCE_DIR CACHE)
  CPMAddPackage(
    NAME llvm_project
    GITHUB_REPOSITORY llvm/llvm-project
    GIT_TAG ${llvm_revision}
    EXCLUDE_FROM_ALL TRUE
    SYSTEM TRUE
    SOURCE_SUBDIR llvm
    GIT_PROGRESS TRUE
    # Required for ninja:
    # https://gitlab.kitware.com/cmake/cmake/-/issues/18238#note_440475.
    USES_TERMINAL_DOWNLOAD TRUE
    OPTIONS "LLVM_ENABLE_PROJECTS mlir\\\\;lld"
    # The interface given by an "In-tree" LLVM build is not identical to the one
    # given by the LLVM Config when using `find_package`. To workaround this,
    # add an external project to the LLVM build that is called from within LLVM
    # via `add_subdirectory`. This gives us the chance to inspect and set the
    # required variables to get both build types to parity.
    "LLVM_EXTERNAL_PROJECTS Pylir"
    "LLVM_EXTERNAL_PYLIR_SOURCE_DIR ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/LLVM-Unified-Adaptor"
    "LLVM_INCLUDE_TESTING OFF"
    "LLVM_INCLUDE_RUNTIMES OFF"
    "LLVM_INCLUDE_EXAMPLES OFF"
    "LLVM_INCLUDE_BENCHMARKS OFF"
    "LLVM_INCLUDE_DOCS OFF"
    # Targets currently tested with Pylir
    # TODO: Make this an option that is automatically quoted.
    "LLVM_TARGETS_TO_BUILD X86\\\\;AArch64"
    "CMAKE_BUILD_TYPE ${PYLIR_LLVM_CMAKE_BUILD_TYPE}"
    "LLVM_ENABLE_RTTI ${PYLIR_ENABLE_RTTI}"
  )

  # LLVMs cmake file currently has a bug where it sets `EXCLUDE_FROM_ALL` to
  # `OFF` within a macro(!), affecting all targets created afterwards.
  # Workaround this by manually going over all targets and explicitly excluding
  # them again.
  macro(exclude_all_targets_recursive dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach (subdir ${subdirectories})
      exclude_all_targets_recursive(${subdir})
    endforeach ()

    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    foreach (target IN LISTS current_targets)
      set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL "ON")
    endforeach ()
  endmacro()

  exclude_all_targets_recursive("${llvm_project_SOURCE_DIR}/llvm")
  exclude_all_targets_recursive("${llvm_project_SOURCE_DIR}/mlir")
  exclude_all_targets_recursive("${llvm_project_SOURCE_DIR}/lld")

  # Fetch the variables set by `LLVM-Unified-Adaptor` cmake from the global
  # property and apply them to this scope as `find_package(LLVM)` would.
  get_property(propagated_flags GLOBAL PROPERTY PYLIR_PROPAGATED_LLVM_FLAGS)
  foreach (pair IN LISTS propagated_flags)
    # A list element has the form "var=value" where `value` has all `;`
    # replaced with `,`.
    string(REGEX MATCH "^[^ =]+" var_key "${pair}")
    string(LENGTH "${pair}" var_length)
    string(LENGTH "${var_key}" var_key_length)
    math(EXPR var_key_length "${var_key_length}+1")
    string(SUBSTRING "${pair}" "${var_key_length}" "-1" var_value)
    string(REPLACE "," ";" var_value "${var_value}")
    # Set the variable in the caller.
    set(${var_key} ${var_value} PARENT_SCOPE)
  endforeach ()
endfunction()
