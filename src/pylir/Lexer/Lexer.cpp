#include "Lexer.hpp"

#include <iterator>

pylir::Lexer::Lexer(std::string_view source, int fieldId) : m_fileId(fieldId), m_source(source)
{
    auto encoding = Text::readBOM(m_source);
    m_encoding = encoding.value_or(Text::Encoding::UTF8);
    m_transcoder.emplace(m_source, m_encoding);
    m_current = m_transcoder->begin();
}

bool pylir::Lexer::parseNext()
{
    if (m_current == m_transcoder->end())
    {
        return false;
    }
    do
    {
        switch (*m_current)
        {
            case U'#':
            {
                m_current = std::find_if(m_current, m_transcoder->end(),
                                         [](char32_t value) { return value == '\n' || value == '\r'; });
                if (m_current == m_transcoder->end())
                {
                    return false;
                }
                if (*m_current == '\r' && std::next(m_current) != m_transcoder->end()
                    && *std::next(m_current) == '\n')
                {
                    m_current++;
                }
                continue;
            }
        }
        break;
    } while (true);
    return true;
}
