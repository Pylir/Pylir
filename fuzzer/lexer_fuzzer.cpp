
#include <pylir/Lexer/Lexer.hpp>

#include <cstdint>
#include <cstring>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size)
{
    std::string input(size, '\0');
    std::memcpy(input.data(), data, size);

    pylir::Diag::Document document(input);
    pylir::Lexer lexer(document);
    std::for_each(lexer.begin(), lexer.end(), [](auto&&) {});
    return 0;
}
