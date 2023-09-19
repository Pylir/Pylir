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

find_package(Threads REQUIRED)
link_libraries(Threads::Threads)
find_package(MLIR REQUIRED CONFIG)
find_package(LLD REQUIRED)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/.pinned-llvm-revision PYLIR_REQUIRED_LLVM_REVISION)
string(STRIP ${PYLIR_REQUIRED_LLVM_REVISION} PYLIR_REQUIRED_LLVM_REVISION)

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

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
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
# Include directory where the various tablegen utilities place auto generated
# sources and headers.
include_directories(SYSTEM ${PROJECT_BINARY_DIR}/src)
include_directories(SYSTEM 3rdParty)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
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

