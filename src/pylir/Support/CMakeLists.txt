
add_library(PylirSupport Text.cpp Endian.hpp TextData.cpp BigInt.cpp)
target_link_libraries(PylirSupport PRIVATE utf8proc PUBLIC tommath)
if (PYLIR_USE_PCH)
    target_link_libraries(PylirSupport PRIVATE COMMON_PCH)
endif ()
if (PYLIR_SANITIZER OR PYLIR_COVERAGE)
    target_compile_options(PylirSupport PRIVATE -fno-sanitize=all -fno-profile-instr-generate -fno-coverage-mapping)
endif ()
