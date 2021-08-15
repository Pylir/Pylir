
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/pylir/${PROJECT_VERSION_MAJOR}")

add_library(PylirRuntime STATIC PylirInteger.cpp PylirGC.cpp)

add_library(PylirRuntimeMain OBJECT PylirRuntimeMain.cpp)
