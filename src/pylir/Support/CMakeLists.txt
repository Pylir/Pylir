
add_library(Support Text.cpp Endian.hpp)
target_link_libraries(Support PRIVATE LLVMSupport PRIVATE utf8proc)
