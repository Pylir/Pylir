
add_library(Support Text.cpp Endian.hpp TextData.cpp)
target_link_libraries(Support PRIVATE LLVMSupport PRIVATE utf8proc)
if (PYLIR_USE_PCH)
    target_link_libraries(Support PRIVATE COMMON_PCH)
endif ()
