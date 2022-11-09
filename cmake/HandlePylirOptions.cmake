
include(CheckCXXCompilerFlag)

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
    add_compile_options(/bigobj /permissive- /W4 /Zc:__cplusplus /utf-8 /EHsc)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
    if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(/Zc:preprocessor)
    endif ()
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?Clang")
    # Flags for both clang-cl and normal clang.
    add_compile_options(-Wno-nullability-completeness -Wno-nullability-extension -Wno-assume
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type-c-linkage>)
endif ()


if (PYLIR_ENABLE_ASSERTIONS)
    # On non-Debug builds cmake automatically defines NDEBUG, so we
    # explicitly undefine it:
    if( NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
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
                string (REGEX REPLACE "(^| )[/-]D *NDEBUG($| )" " "
                        "${flags_var_to_scrub}" "${${flags_var_to_scrub}}")
            endforeach()
        endif()
    endif()
endif ()

# Clang-cl is not used for linking, cmake calls lld-link directly. We have to pass the runtime directory of clang-cl
# instead to find the directory where the runtime libraries are contained in.
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

# Pass -Wl,-z,defs. This makes sure all symbols are defined. Otherwise a DSO
# build might work on ELF but fail on MachO/COFF.
if (NOT (CMAKE_SYSTEM_NAME MATCHES "Darwin|FreeBSD|OpenBSD|DragonFly|AIX|OS390" OR
        WIN32 OR CYGWIN) AND
        NOT PYLIR_SANITIZER)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,defs")
endif ()

# Matching LLVMs visibility option here. Mismatch of visibility can cause linker warnings on macOS.
if((NOT (${CMAKE_SYSTEM_NAME} MATCHES "AIX")) AND
(NOT (WIN32 OR CYGWIN) OR (MINGW AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")))
    # GCC for MinGW does nothing about -fvisibility-inlines-hidden, but warns
    # about use of the attributes. As long as we don't use the attributes (to
    # override the default) we shouldn't set the command line options either.
    # GCC on AIX warns if -fvisibility-inlines-hidden is used and Clang on AIX doesn't currently support visibility.
    check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
    if (SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
    endif ()
endif()

if (POLICY CMP0116)
    # TODO: Reevaluate once minimum version is 3.20. Affects the way depfiles are handled in cmake. Setting it to OLD
    #       makes it behave the same way in all versions.
    cmake_policy(SET CMP0116 OLD)
endif ()

# Matching LLVMs RTTI setting here. Could cause linker issues otherwise.
if (NOT LLVM_ENABLE_RTTI)
    if (MSVC)
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-")
    else ()
        string(REGEX REPLACE "-frtti" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
    endif ()
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
endif()