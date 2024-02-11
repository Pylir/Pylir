# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CPM)

CPMAddPackage("gh:catchorg/Catch2@3.5.2")
CPMAddPackage("gh:fmtlib/fmt#10.2.1")
# Use the same C++ version everywhere.
set(CTRE_CXX_STANDARD ${CMAKE_CXX_STANDARD})
CPMAddPackage("gh:hanickadot/compile-time-regular-expressions#v3.8")
CPMAddPackage(
  NAME libtommath
  GIT_TAG 8314bde5e5c8e5d9331460130a9d1066e324f091
  GITHUB_REPOSITORY libtom/libtommath
  DOWNLOAD_ONLY YES
  EXCLUDE_FROM_ALL
  SYSTEM
)
CPMAddPackage(
  NAME utf8proc
  VERSION 2.9.0
  GITHUB_REPOSITORY JuliaStrings/utf8proc
  DOWNLOAD_ONLY YES
  EXCLUDE_FROM_ALL
  SYSTEM
)

#===------------------------------------------------------------------------===#
# utf8proc
#===------------------------------------------------------------------------===#

add_library(utf8proc STATIC EXCLUDE_FROM_ALL ${utf8proc_SOURCE_DIR}/utf8proc.c)
target_compile_definitions(utf8proc PUBLIC UTF8PROC_STATIC)
target_include_directories(utf8proc SYSTEM PUBLIC ${utf8proc_SOURCE_DIR})

#===------------------------------------------------------------------------===#
# libtommath
#===------------------------------------------------------------------------===#

file(GLOB tommath_srcs "${libtommath_SOURCE_DIR}/*.c")
add_library(tommath STATIC EXCLUDE_FROM_ALL ${tommath_srcs})
target_include_directories(tommath SYSTEM PUBLIC ${libtommath_SOURCE_DIR})
target_compile_definitions(tommath
  PUBLIC MP_NO_ZERO_ON_FREE MP_NO_FILE MP_FIXED_CUTOFFS)

#===------------------------------------------------------------------------===#
# Catch2
#===------------------------------------------------------------------------===#

# Workaround for https://github.com/cpm-cmake/CPM.cmake/issues/475.
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

#===------------------------------------------------------------------------===#
# libunwind
#===------------------------------------------------------------------------===#

if (MSVC)
  # Workaround for cmake issue:
  # https://gitlab.kitware.com/cmake/cmake/-/commit/b41fab928551d5a71af2c16b34f9683d872818b0
  # TODO: Remove below once CMake 3.24 or higher is the minimum required version.
  set(CMAKE_ASM-ATT_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded "")
  set(CMAKE_ASM-ATT_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL "")
  set(CMAKE_ASM-ATT_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug "")
  set(CMAKE_ASM-ATT_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL "")
  set(LIBUNWIND_ENABLE_SHARED OFF)
  add_compile_definitions(__SEH__)
  add_compile_options(-fno-sanitize=all -fno-profile-instr-generate -fno-coverage-mapping)
  add_subdirectory(libunwind EXCLUDE_FROM_ALL)
endif ()

# Propagate module path set by 3rd party libraries upwards.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
