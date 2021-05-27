#include "Lexer.hpp"

#include <iterator>

pylir::Lexer::Lexer(std::string_view source, int fieldId) : m_fileId(fieldId), m_source(source)
{
    auto encoding = Text::readBOM(m_source);
    m_encoding = encoding.value_or(Text::Encoding::UTF8);
    m_transcoder.emplace(m_source, m_encoding);
    m_current = m_transcoder->begin();
}

namespace
{
template <class T>
auto advancePastNewline(T& iterator, T end)
{
    if (*iterator == '\r' && std::next(iterator) != end && *std::next(iterator) == '\n')
    {
        std::advance(iterator, 2);
    }
    else
    {
        std::advance(iterator, 1);
    }
}
} // namespace

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
                    break;
                }
            }
                [[fallthrough]];
            case U'\r':
            case U'\n':
            {
                auto offset = m_current - m_transcoder->begin();
                advancePastNewline(m_current, m_transcoder->end());
                m_tokens.emplace_back(offset, m_current - m_transcoder->begin() - offset, m_fileId, TokenType::Newline);
                m_lineStarts.push_back(m_current - m_transcoder->begin());
                break;
            }
            case U'\\':
            {
                m_current++;
                if (m_current == m_transcoder->end())
                {
                    // TODO: Unexpected end of file
                    return false;
                }
                if (*m_current != U'\n' && *m_current != U'\r')
                {
                    // TODO: Expected newline after line continuation
                    return false;
                }
                advancePastNewline(m_current, m_transcoder->end());
                m_lineStarts.push_back(m_current - m_transcoder->begin());
                break;
            }
        }
        break;
    } while (true);
    if (m_current == m_transcoder->end())
    {
        m_tokens.emplace_back(m_current - m_transcoder->begin(), 0, m_fileId, TokenType::Newline);
        m_lineStarts.push_back(m_current - m_transcoder->begin());
    }
    return true;
}
