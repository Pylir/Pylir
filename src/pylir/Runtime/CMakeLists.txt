
if (CMAKE_CXX_COMPILER_TARGET)
    set(TARGET_TRIPLE ${CMAKE_CXX_COMPILER_TARGET})
endif ()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/pylir/${TARGET_TRIPLE}")

function(get_sources target listvar sourcevar definevar includevar)
    set(worklist ${${listvar}})
    set(sourcelist ${${sourcevar}})
    set(definelist ${${definevar}})
    set(includelist ${${includevar}})
    if (TARGET ${target})
        list(APPEND worklist ${target})
        get_target_property(targetSources ${target} SOURCES)
        if (targetSources STREQUAL "targetSources-NOTFOUND")
            return()
        endif ()
        get_target_property(targetSourceDir ${target} SOURCE_DIR)
        list(TRANSFORM targetSources PREPEND "${targetSourceDir}/")
        list(APPEND sourcelist ${targetSources})
        get_target_property(targetCompileDefs ${target} INTERFACE_COMPILE_DEFINITIONS)
        if (NOT targetCompileDefs STREQUAL "targetCompileDefs-NOTFOUND")
            list(APPEND definelist ${targetCompileDefs})
        endif ()
        get_target_property(targetIncludeDir ${target} INTERFACE_INCLUDE_DIRECTORIES)
        if (NOT targetIncludeDir STREQUAL "targetIncludeDir-NOTFOUND")
            list(APPEND includelist ${targetIncludeDir})
        endif ()
        get_target_property(subDeps ${target} LINK_LIBRARIES)
        foreach (dep IN LISTS subDeps)
            if (NOT dep IN_LIST worklist)
                get_sources(${dep} worklist sourcelist definelist includelist)
            endif ()
        endforeach ()
        set(${listvar} "${worklist}" PARENT_SCOPE)
        set(${sourcevar} "${sourcelist}" PARENT_SCOPE)
        set(${definevar} "${definelist}" PARENT_SCOPE)
        set(${includevar} "${includelist}" PARENT_SCOPE)
    endif ()
endfunction()

get_sources(PylirSupport _ SOURCES DEFINES INCLUDES)

add_library(PylirRuntime STATIC Stack.cpp Objects.cpp API.cpp ExceptionHandling.cpp Support.cpp SysModule.cpp ${SOURCES} Pages.cpp Pages.hpp Globals.cpp)
target_include_directories(PylirRuntime PUBLIC ${INCLUDES})
target_compile_definitions(PylirRuntime PUBLIC ${DEFINES})
if (PYLIR_SANITIZER OR PYLIR_COVERAGE)
    target_compile_options(PylirRuntime PUBLIC -fno-sanitize=all -fno-profile-instr-generate -fno-coverage-mapping)
endif ()

add_subdirectory(MarkAndSweep)

if (MSVC)
    add_custom_command(TARGET PylirRuntime POST_BUILD
            COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/util/merge_archives.py
            ARGS
            -o "$<TARGET_LINKER_FILE:PylirRuntime>"
            --ar "${CMAKE_AR}"
            --msvc
            ${MERGE_ARCHIVES_LIBTOOL}
            "$<TARGET_LINKER_FILE:PylirRuntime>"
            "$<TARGET_LINKER_FILE:unwind_static>")
    add_dependencies(PylirRuntime unwind_static)
elseif (UNIX)
    find_path(LIBUNWIND_INCLUDE_DIR libunwind.h REQUIRED)
    find_library(LIBUNWIND_LIB unwind REQUIRED)
    target_link_libraries(PylirRuntime PRIVATE ${LIBUNWIND_LIB})
    target_include_directories(PylirRuntime PRIVATE ${LIBUNWIND_INCLUDE_DIR})
endif ()

add_library(PylirRuntimeMain STATIC PylirRuntimeMain.cpp)
if (PYLIR_SANITIZER OR PYLIR_COVERAGE)
    target_compile_options(PylirRuntimeMain PRIVATE -fno-sanitize=all -fno-profile-instr-generate -fno-coverage-mapping)
endif ()
