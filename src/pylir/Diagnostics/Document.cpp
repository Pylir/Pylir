#include "Document.hpp"

pylir::Diag::Document::Document(std::string input, std::string filename, pylir::Text::Encoding encoding)
    : m_filename(std::move(filename)), m_input(std::move(input))
{
    std::string_view view = m_input;
    m_encoding = Text::readBOM(view).value_or(encoding);
    m_text.reserve(view.size());
    auto transcoder = Text::Transcoder<void, char32_t>(view, m_encoding);
    for (auto iter = transcoder.begin(); iter != transcoder.end();)
    {
        switch (*iter)
        {
            case '\r':
            {
                std::size_t increment = 1;
                if (std::next(iter) != transcoder.end() && *std::next(iter) == '\n')
                {
                    increment = 2;
                }
                m_text += '\n';
                m_lineStarts.push_back(m_text.size());
                std::advance(iter, increment);
                break;
            }
            case '\n': m_lineStarts.push_back(m_text.size() + 1); [[fallthrough]];
            default: m_text += *iter++;
        }
    }
    // + 1 for imaginary newline that does not exist
    m_lineStarts.push_back(m_text.size() + 1);
}
