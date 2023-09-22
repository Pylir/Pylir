# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.20)
project(Pylir VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)

option(PYLIR_BUILD_TESTS "Build tests" ON)
option(PYLIR_BUILD_DOCS "Build documentation" OFF)
option(PYLIR_FUZZER "Build fuzzers" OFF)
option(PYLIR_COVERAGE "Compile with coverage" OFF)
option(PYLIR_INCLUDE_LLVM_BUILD
  "Whether to download and build LLVM as part of Pylir" ON)
set(PYLIR_SANITIZERS "" CACHE STRING "Compile with given sanitizers")

if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  option(PYLIR_ENABLE_ASSERTIONS "Enable assertions" OFF)
else ()
  option(PYLIR_ENABLE_ASSERTIONS "Enable assertions" ON)
endif ()

## Set up dependencies and global include and link directories.

# All platforms we currently care about default to PIC. This is also required
# when linking a shared library (including a static into a shared).
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE "BOOL" "")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# NO_POLICY_SCOPE makes policy changes done in the include affect the caller.
include(CMakePolicies NO_POLICY_SCOPE)

add_subdirectory(3rdParty)

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/.pinned-llvm-revision PYLIR_REQUIRED_LLVM_REVISION)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/.pinned-llvm-revision)
string(STRIP ${PYLIR_REQUIRED_LLVM_REVISION} PYLIR_REQUIRED_LLVM_REVISION)

if (PYLIR_INCLUDE_LLVM_BUILD)
  include(CPM)
  
  # Default LLVM options to corresponding Pylir options.
  set(PYLIR_LLVM_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
    CACHE STRING "Build type to use for LLVM compilation")
  option(LLVM_ENABLE_ASSERTIONS "Whether to build LLVM with assertions"
    ${PYLIR_ENABLE_ASSERTIONS})
  
  string(REPLACE "address" "Address" llvm_sanitizer_default
    "${PYLIR_SANITIZERS}")
  string(REPLACE "undefined" "Undefined" llvm_sanitizer_default
    "${llvm_sanitizer_default}")
  string(REPLACE "thread" "Thread" llvm_sanitizer_default
    "${llvm_sanitizer_default}")
  string(REPLACE "," ";" llvm_sanitizer_default "${llvm_sanitizer_default}")
  
  set(LLVM_USE_SANITIZER "${llvm_sanitizer_default}" CACHE
    STRING "Sanitizers to use when building LLVM")
  
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
    GIT_TAG ${PYLIR_REQUIRED_LLVM_REVISION}
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
    "LLVM_EXTERNAL_PYLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake/LLVM-Unified-Adaptor"
    "LLVM_INCLUDE_TESTING OFF"
    "LLVM_INCLUDE_BENCHMARKS OFF"
    "LLVM_INCLUDE_DOCS OFF"
    # Targets currently tested with Pylir
    # TODO: Make this an option that is automatically quoted.
    "LLVM_TARGETS_TO_BUILD X86\\\\;AArch64"
    "CMAKE_BUILD_TYPE ${PYLIR_LLVM_CMAKE_BUILD_TYPE}"
    "LLVM_USE_SANITIZER ${LLVM_USE_SANITIZER}"
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
    set(${var_key} ${var_value})
  endforeach ()

else ()
  find_package(Threads REQUIRED)
  link_libraries(Threads::Threads)
  find_package(MLIR REQUIRED CONFIG)
  find_package(LLD REQUIRED)
  
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  
  find_file(VCS_HEADER VCSRevision.h PATHS ${LLVM_INCLUDE_DIRS} PATH_SUFFIXES llvm/Support/ REQUIRED NO_DEFAULT_PATH)
  message(STATUS "Checking ${VCS_HEADER} for version mismatch")
  file(READ ${VCS_HEADER} VCS_FILE)
  string(REGEX MATCH "#define LLVM_REVISION \"([a-z0-9]+)\"" VCS_MATCH ${VCS_FILE})
  if (CMAKE_MATCH_COUNT EQUAL 0)
    message(WARNING "Failed to determine revision of LLVM installed. Proceed with caution.
Required revision: ${PYLIR_REQUIRED_LLVM_REVISION}
Found LLVM installation: ${MLIR_DIR}")
  else ()
    if (NOT ${CMAKE_MATCH_1} STREQUAL PYLIR_REQUIRED_LLVM_REVISION)
      message(WARNING "Installed LLVM revision (${CMAKE_MATCH_1}) does not match required revision.\
        Compilation is likely to fail.
Required revision: ${PYLIR_REQUIRED_LLVM_REVISION}
Found LLVM installation: ${MLIR_DIR}")
    endif ()
  endif ()
  
  link_directories(${LLVM_BUILD_LIBRARY_DIR})
endif ()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(HandlePylirOptions)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddPylir)

set(LLVM_TABLEGEN_EXE "llvm-tblgen")
set(PYLIR_TABLEGEN_EXE "pylir-tblgen")

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${LLD_INCLUDE_DIRS})
# Include directory where the various tablegen utilities place auto generated
# sources and headers.
include_directories(SYSTEM ${PROJECT_BINARY_DIR}/src)
include_directories(SYSTEM 3rdParty)
add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(PYLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PYLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PYLIR_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

# Required by tests and for some build utilities.
find_package(Python3 3.6 COMPONENTS Interpreter REQUIRED)

if (PYLIR_BUILD_DOCS)
  add_subdirectory(docs)
endif ()

if (PYLIR_FUZZER)
  add_subdirectory(fuzzer)
endif ()

add_subdirectory(src)
add_subdirectory(tools)
if (PYLIR_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(unittests)
  add_subdirectory(test)
endif ()

