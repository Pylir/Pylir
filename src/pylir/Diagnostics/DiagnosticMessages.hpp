
#pragma once

#include <string_view>

#include <fmt/format.h>

#ifdef NDEBUG
    #define FORMAT_STRING(...) \
        std::string_view       \
        {                      \
            __VA_ARGS__        \
        }
#else
    #define FORMAT_STRING(...) FMT_STRING(__VA_ARGS__)
#endif

namespace pylir::Diag
{
constexpr auto UNEXPECTED_EOF_WHILE_PARSING = FORMAT_STRING("unexpected EOF while parsing\n"
                                                            "\n"
                                                            "{0:line-1}\n"
                                                            "{0:line,coloured{1}}\n"
                                                            "{1:^}\n"
                                                            "{1:\\n}\n"
                                                            "{0:line+1}\n"
                                                            "\n");
}
