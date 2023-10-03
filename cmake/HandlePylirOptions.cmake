# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CheckCXXCompilerFlag)

# Macro consolidating all logic of compile options for Pylir but not any of its
# dependencies.
macro(add_project_compile_options)
  if (NOT MSVC)
    # GNU style flags.
    add_compile_options(-pedantic -Wall -Wextra $<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      if (WIN32)
        if (NOT ${CMAKE_CXX_FLAGS} MATCHES ".*[ \t\r\n]-flto[^a-zA-Z_].*")
          add_compile_options(-Wa,-mbig-obj)
        endif ()
      endif ()
    endif ()
  elseif (MSVC)
    # MSVC style flags.
    add_compile_options(/bigobj /permissive- /W4 /Zc:__cplusplus /utf-8)
    if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      add_compile_options(/Zc:preprocessor)
    endif ()
  endif ()

  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?Clang")
    # Flags for both clang-cl and normal clang.
    add_compile_options(-Wno-nullability-completeness -Wno-nullability-extension -Wno-assume
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type-c-linkage>)
  endif ()
endmacro()

# Macro consolidating all logic of compile options for Pylir and all of its
# dependencies. This mostly contains compiler instrumentation options and code
# generation options that have to match between all dependencies.
macro(add_global_compile_options)
  # All platforms we currently care about default to PIC. This is also required
  # when linking a shared library (including a static into a shared).
  set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE "BOOL" "")

  # Workaround https://github.com/llvm/llvm-project/issues/65255
  if (MSVC)
    add_compile_options(/EHsc)
    add_compile_definitions(_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING
      _CRT_SECURE_NO_WARNINGS)
  endif ()

  if (PYLIR_ENABLE_ASSERTIONS)
    # On non-Debug builds cmake automatically defines NDEBUG, so we
    # explicitly undefine it:
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
      # NOTE: use `add_compile_options` rather than `add_definitions` since
      # `add_definitions` does not support generator expressions.
      add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-UNDEBUG>)
      if (MSVC)
        # Also remove /D NDEBUG to avoid MSVC warnings about conflicting defines.
        foreach (flags_var_to_scrub
          CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_RELWITHDEBINFO
          CMAKE_CXX_FLAGS_MINSIZEREL
          CMAKE_C_FLAGS_RELEASE
          CMAKE_C_FLAGS_RELWITHDEBINFO
          CMAKE_C_FLAGS_MINSIZEREL)
          string(REGEX REPLACE "(^| )[/-]D *NDEBUG($| )" " "
            "${flags_var_to_scrub}" "${${flags_var_to_scrub}}")
        endforeach ()
      endif ()
    endif ()
  endif ()

  # Clang-cl is not used for linking, cmake calls lld-link directly. We have to
  # pass the runtime directory of clang-cl instead to find the directory where
  # the runtime libraries are contained in.
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND MSVC AND (PYLIR_COVERAGE OR PYLIR_SANITIZERS OR PYLIR_FUZZER))
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

  if (PYLIR_SANITIZERS)
    if (MSVC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oy- -fsanitize=${PYLIR_SANITIZERS} -fno-sanitize-recover=all")
      link_libraries(clang_rt.asan.lib)
      link_libraries(clang_rt.asan_cxx.lib)
      link_libraries(clang_rt.asan-preinit.lib)
    else ()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=${PYLIR_SANITIZERS} -fno-sanitize-recover=all")
    endif ()
  endif ()

  # Pass -Wl,-z,defs. This makes sure all symbols are defined. Otherwise a DSO
  # build might work on ELF but fail on MachO/COFF.
  if (NOT (CMAKE_SYSTEM_NAME MATCHES "Darwin|FreeBSD|OpenBSD|DragonFly|AIX|OS390" OR
    WIN32 OR CYGWIN) AND
    NOT PYLIR_SANITIZERS)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,defs")
  endif ()

  # Matching LLVMs visibility option here. Mismatch of visibility can cause linker warnings on macOS.
  if ((NOT (${CMAKE_SYSTEM_NAME} MATCHES "AIX")) AND
  (NOT (WIN32 OR CYGWIN) OR (MINGW AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")))
    # GCC for MinGW does nothing about -fvisibility-inlines-hidden, but warns
    # about use of the attributes. As long as we don't use the attributes (to
    # override the default) we shouldn't set the command line options either.
    # GCC on AIX warns if -fvisibility-inlines-hidden is used and Clang on AIX doesn't currently support visibility.
    check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
    if (SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
    endif ()
  endif ()

  set(to_replace)
  set(to_add)
  if (MSVC)
    set(to_replace "/GR-")
    set(to_add "/GR")
  else ()
    set(to_replace "-fno-rtti")
    set(to_add "-frtti")
  endif ()
  if (NOT PYLIR_ENABLE_RTTI)
    set(temp ${to_replace})
    set(to_replace ${to_add})
    set(to_add ${temp})
  endif ()
  string(REGEX REPLACE "${to_replace}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${to_add}")

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
  endif ()
endmacro()
