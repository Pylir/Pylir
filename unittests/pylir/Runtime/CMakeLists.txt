
include(PylirCompile)

pylir_obj_compile(builtins.o empty.py)
add_library(PylirTestRuntime STATIC builtins.o)
target_link_libraries(PylirTestRuntime PUBLIC PylirRuntime)

add_subdirectory(MarkAndSweep)
