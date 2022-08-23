# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.15)
project(Pylir VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)

option(PYLIR_BUILD_TESTS "Build tests" ON)
option(PYLIR_FUZZER "Build fuzzers" OFF)
option(PYLIR_COVERAGE "Compile with coverage" OFF)
option(PYLIR_EMBED_LLD "Embed lld into pylir" OFF)
set(PYLIR_DEFAULT_SYSROOT "" CACHE STRING "Default sysroot to find system libraries")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND MSVC AND (PYLIR_COVERAGE OR PYLIR_SANITIZER OR PYLIR_FUZZER))
    execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} /clang:-print-libgcc-file-name /clang:--rtlib=compiler-rt
            OUTPUT_VARIABLE clang_compiler_rt_file
            ERROR_VARIABLE clang_cl_stderr
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE clang_cl_exit_code)
    if (NOT "${clang_cl_exit_code}" STREQUAL "0")
        message(FATAL_ERROR
                "Unable to invoke clang-cl to find resource dir: ${clang_cl_stderr}")
    endif ()
    file(TO_CMAKE_PATH "${clang_compiler_rt_file}" clang_compiler_rt_file)
    get_filename_component(clang_runtime_dir "${clang_compiler_rt_file}" DIRECTORY)
    message(STATUS "Clang-cl runtimes found in ${clang_runtime_dir}")
    link_directories(${clang_runtime_dir})
endif ()

if (PYLIR_COVERAGE)
    message(STATUS "Compiling with Coverage")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options(--coverage)
        if (WIN32)
            link_libraries(gcov)
        endif ()
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
        if (NOT MSVC)
            add_link_options(-fprofile-instr-generate)
        endif ()
    else ()
        message(ERROR "Unknown coverage implementation")
    endif ()
endif ()

if (DEFINED PYLIR_SANITIZER)
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oy- -fsanitize=${PYLIR_SANITIZER} -fno-sanitize-recover=all")
        link_libraries(clang_rt.asan.lib)
        link_libraries(clang_rt.asan_cxx.lib)
        link_libraries(clang_rt.asan-preinit.lib)
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=${PYLIR_SANITIZER} -fno-sanitize-recover=all")
    endif ()
endif ()

if (PYLIR_USE_PCH)
    add_library(COMMON_PCH INTERFACE)
    target_precompile_headers(COMMON_PCH INTERFACE <type_traits> <functional> <tuple> <optional> <variant> <memory> <string> <string_view> <charconv> <array> <vector> <set> <map> <unordered_set> <unordered_map> <iterator> <algorithm> <numeric> <tcb/span.hpp>)
endif ()

if (POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif ()

find_package(Threads REQUIRED)
link_libraries(Threads::Threads)
# revision 74b5dad5e944db21e56e6d81eb14f11891a741ab
find_package(MLIR REQUIRED CONFIG)

if (PYLIR_EMBED_LLD)
    find_package(LLD REQUIRED)
endif ()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(LLVM_TABLEGEN_EXE "llvm-tblgen")

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/src)
include_directories(SYSTEM ${PROJECT_BINARY_DIR}/src)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(PYLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PYLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PYLIR_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

set(PYLIR_TABLEGEN_EXE "pylir-tblgen")

if (NOT LLVM_ENABLE_RTTI)
    if (MSVC)
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-")
    else ()
        string(REGEX REPLACE "-frtti" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
    endif ()
endif ()

if (NOT MSVC)
    add_compile_options(-pedantic -Wall -Wextra $<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if (WIN32)
            if (NOT ${CMAKE_CXX_FLAGS} MATCHES ".*[ \t\r\n]-flto[^a-zA-Z_].*")
                add_compile_options(-Wa,-mbig-obj)
            endif ()
        endif ()
    endif ()
elseif (MSVC)
    add_compile_options(/bigobj /permissive- /W4 /Zc:__cplusplus /utf-8 /EHsc)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
    if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(/Zc:preprocessor)
    endif ()
endif ()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-Wno-nullability-completeness -Wno-nullability-extension -Wno-assume)
endif ()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Python COMPONENTS Interpreter REQUIRED)

include_directories(src)
include_directories(SYSTEM 3rdParty)
add_subdirectory(3rdParty)
add_subdirectory(src)
add_subdirectory(tools)
if (PYLIR_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(unittests)
    add_subdirectory(test)
endif ()

if (PYLIR_FUZZER)
    add_compile_definitions(PYLIR_IN_FUZZER)
    if (WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer-no-link,address -fno-sanitize-recover=all")
        link_libraries(clang_rt.asan.lib)
        link_libraries(clang_rt.asan_cxx.lib)
        link_libraries(clang_rt.asan-preinit.lib)
        link_libraries(clang_rt.fuzzer.lib)
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=fuzzer-no-link,undefined,address -fno-sanitize-recover=all")
    endif ()

    add_subdirectory(fuzzer)
endif ()
