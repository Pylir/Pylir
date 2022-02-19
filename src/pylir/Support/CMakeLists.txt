
add_library(PylirSupport Text.cpp TextData.cpp BigInt.cpp)
target_link_libraries(PylirSupport PRIVATE utf8proc PUBLIC tommath)
if (PYLIR_USE_PCH)
    target_link_libraries(PylirSupport PRIVATE COMMON_PCH)
endif ()
