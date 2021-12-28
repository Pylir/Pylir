
if (CMAKE_CXX_COMPILER_TARGET)
    set(TARGET_TRIPLE ${CMAKE_CXX_COMPILER_TARGET})
endif ()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/pylir/${TARGET_TRIPLE}")

add_library(PylirRuntime STATIC PylirGC.cpp Objects.cpp API.cpp)
target_link_libraries(PylirRuntime PUBLIC PylirSupport)
if (PYLIR_SANITIZER)
    target_compile_options(PylirRuntime PRIVATE -fno-sanitize=all)
endif ()

if (APPLE)
    set(MERGE_ARCHIVES_LIBTOOL "--use-libtool" "--libtool" "${CMAKE_LIBTOOL}")
endif ()

if (MSVC)
    set(MSVC_FLAG "--msvc")
endif ()

add_custom_command(TARGET PylirRuntime POST_BUILD
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/util/merge_archives.py
        ARGS
        -o "$<TARGET_LINKER_FILE:PylirRuntime>"
        --ar "${CMAKE_AR}"
        ${MSVC_FLAG}
        ${MERGE_ARCHIVES_LIBTOOL}
        "$<TARGET_LINKER_FILE:PylirRuntime>"
        "$<TARGET_LINKER_FILE:PylirSupport>"
        "$<TARGET_LINKER_FILE:utf8proc>"
        "$<TARGET_LINKER_FILE:tommath>"
        DEPENDS PylirSupport utf8proc tommath
        )

add_library(PylirRuntimeMain STATIC PylirRuntimeMain.cpp)
if (PYLIR_SANITIZER)
    target_compile_options(PylirRuntimeMain PRIVATE -fno-sanitize=all)
endif ()
