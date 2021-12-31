
add_library(utf8proc utf8proc.c)
target_compile_definitions(utf8proc PUBLIC UTF8PROC_STATIC)
if (PYLIR_SANITIZER OR PYLIR_COVERAGE)
    target_compile_options(utf8proc PRIVATE -fno-sanitize=all -fno-profile-instr-generate -fno-coverage-mapping)
endif ()
