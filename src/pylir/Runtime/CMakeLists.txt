
add_library(PylirRuntime STATIC PylirInteger.cpp PylirGC.cpp)

add_library(PylirRuntimeMain OBJECT PylirRuntimeMain.cpp)
