
add_library(utf8proc utf8proc.c)
target_compile_definitions(utf8proc PUBLIC UTF8PROC_STATIC)
if (PYLIR_SANITIZER)
    target_compile_options(utf8proc PRIVATE -fno-sanitize=all)
endif ()
