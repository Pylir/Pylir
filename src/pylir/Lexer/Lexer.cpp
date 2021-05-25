#include "Lexer.hpp"

pylir::Lexer::Lexer(std::string_view source, int fieldId) : m_fileId(fieldId), m_source(source)
{
    auto encoding = Text::readBOM(m_source);
    m_encoding = encoding.value_or(Text::Encoding::UTF8);
}

bool pylir::Lexer::parseNext()
{
    return true;
}
