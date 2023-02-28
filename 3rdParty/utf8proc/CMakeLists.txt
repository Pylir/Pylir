
add_library(utf8proc STATIC utf8proc.c)
target_compile_definitions(utf8proc PUBLIC UTF8PROC_STATIC)
