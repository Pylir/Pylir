#include <pylir/Lexer/Lexer.hpp>

#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Expected file as first argument";
        return -1;
    }
    std::ifstream stream(argv[1], std::ios_base::binary);
    if (!stream.is_open())
    {
        std::cerr << "Failed to open " << argv[1];
        return -1;
    }
    std::string result;
    stream.seekg(0, std::ios_base::end);
    std::size_t pos = stream.tellg();
    stream.seekg(0, std::ios_base::beg);
    result.resize(pos, '\0');
    stream.read(result.data(), result.size());

    pylir::Diag::Document document(result);
    pylir::Lexer lexer(document);
    std::for_each(lexer.begin(), lexer.end(), [](auto&&) {});
}
