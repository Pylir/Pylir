
add_library(Lexer Lexer.cpp Token.cpp)
target_link_libraries(Lexer PUBLIC LLVMSupport Support PUBLIC Diagnostics)
if (PYLIR_USE_PCH)
    target_link_libraries(Lexer PRIVATE COMMON_PCH)
    target_precompile_headers(Lexer PRIVATE <fmt/format.h> <llvm/ADT/APInt.h>)
endif ()
