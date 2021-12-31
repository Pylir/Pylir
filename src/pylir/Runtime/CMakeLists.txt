
if (CMAKE_CXX_COMPILER_TARGET)
    set(TARGET_TRIPLE ${CMAKE_CXX_COMPILER_TARGET})
endif ()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/pylir/${TARGET_TRIPLE}")

add_library(PylirRuntime STATIC PylirGC.cpp Objects.cpp API.cpp ExceptionHandling.cpp)
target_link_libraries(PylirRuntime PUBLIC PylirSupport)
if (PYLIR_SANITIZER OR PYLIR_COVERAGE)
    target_compile_options(PylirRuntime PRIVATE -fno-sanitize=all -fno-profile-instr-generate -fno-coverage-mapping)
endif ()

if (APPLE)
    set(MERGE_ARCHIVES_LIBTOOL "--use-libtool" "--libtool" "${CMAKE_LIBTOOL}")
endif ()

if (MSVC)
    add_custom_command(TARGET PylirRuntime POST_BUILD
            COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/util/merge_archives.py
            ARGS
            -o "$<TARGET_LINKER_FILE:PylirRuntime>"
            --ar "${CMAKE_AR}"
            --msvc
            ${MERGE_ARCHIVES_LIBTOOL}
            "$<TARGET_LINKER_FILE:PylirRuntime>"
            "$<TARGET_LINKER_FILE:PylirSupport>"
            "$<TARGET_LINKER_FILE:utf8proc>"
            "$<TARGET_LINKER_FILE:tommath>"
            "$<TARGET_LINKER_FILE:unwind_static>")
    add_dependencies(PylirRuntime PylirSupport utf8proc tommath unwind_static)
else ()
    add_custom_command(TARGET PylirRuntime POST_BUILD
            COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/util/merge_archives.py
            ARGS
            -o "$<TARGET_LINKER_FILE:PylirRuntime>"
            --ar "${CMAKE_AR}"
            ${MERGE_ARCHIVES_LIBTOOL}
            "$<TARGET_LINKER_FILE:PylirRuntime>"
            "$<TARGET_LINKER_FILE:PylirSupport>"
            "$<TARGET_LINKER_FILE:utf8proc>"
            "$<TARGET_LINKER_FILE:tommath>"
            DEPENDS PylirSupport utf8proc tommath
            )
    add_dependencies(PylirRuntime PylirSupport utf8proc tommath)
endif ()


add_library(PylirRuntimeMain STATIC PylirRuntimeMain.cpp)
if (PYLIR_SANITIZER OR PYLIR_COVERAGE)
    target_compile_options(PylirRuntimeMain PRIVATE -fno-sanitize=all -fno-profile-instr-generate -fno-coverage-mapping)
endif ()
