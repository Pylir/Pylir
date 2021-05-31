
#pragma once

#include <pylir/Support/LazyCacheIterator.hpp>
#include <pylir/Support/Text.hpp>

#include <algorithm>
#include <string>
#include <string_view>

#include <tcb/span.hpp>

namespace pylir::Diag
{
class Document
{
    std::string m_filename;
    Text::Encoding m_encoding;
    std::string m_input;
    std::u32string m_text;
    std::vector<std::size_t> m_lineStarts{0};
    Text::Transcoder<void, char32_t> m_transcoder;
    decltype(m_transcoder)::iterator m_current;

    bool checkNext()
    {
        if (m_current == m_transcoder.end())
        {
            return false;
        }
        switch (*m_current)
        {
            case '\r':
            {
                std::size_t increment = 1;
                if (std::next(m_current) != m_transcoder.end() && *std::next(m_current) == '\n')
                {
                    increment = 2;
                }
                m_text += '\n';
                m_lineStarts.push_back(m_text.size());
                std::advance(m_current, increment);
                return true;
            }
            case '\n': m_lineStarts.push_back(m_text.size() + 1); [[fallthrough]];
            default: m_text += *m_current++;
        }
        if (m_current == m_transcoder.end())
        {
            // + 1 for imaginary newline that does not exist
            m_lineStarts.push_back(m_text.size() + 1);
        }
        return true;
    }

public:
    using value_type = decltype(m_transcoder)::value_type;
    using reference = decltype(m_transcoder)::reference;
    using const_reference = decltype(m_transcoder)::const_reference;
    using iterator = LazyCacheIterator<value_type, Document, &Document::checkNext, &Document::m_text>;
    using const_iterator = iterator;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    Document(std::string input, std::string filename = "<stdin>", Text::Encoding encoding = Text::Encoding::UTF8)
        : m_filename(std::move(filename)),
          m_encoding(Text::checkForBOM(input).value_or(encoding)),
          m_input(std::move(input)),
          m_transcoder(
              [this]
              {
                  std::string_view view = m_input;
                  Text::readBOM(view);
                  return view;
              }(),
              m_encoding)
    {
        m_current = m_transcoder.begin();
        m_text.reserve(input.size());
    }

    iterator begin()
    {
        return iterator(*this, 0);
    }

    const_iterator cbegin()
    {
        return begin();
    }

    iterator end()
    {
        return iterator(*this, -1);
    }

    const_iterator cend()
    {
        return end();
    }

    std::size_t getLineNumber(std::size_t offset) const
    {
        auto result = std::lower_bound(m_lineStarts.begin(), m_lineStarts.end(), offset);
        return result - m_lineStarts.begin();
    }

    std::size_t getColNumber(std::size_t offset) const
    {
        auto lineNumber = getLineNumber(offset);
        return offset - m_lineStarts[lineNumber - 1] + 1;
    }

    std::pair<std::size_t, std::size_t> getLineCol(std::size_t offset) const
    {
        auto lineNumber = getLineNumber(offset);
        return {lineNumber, offset - m_lineStarts[lineNumber - 1] + 1};
    }

    std::u32string_view getLine(std::size_t lineNumber) const
    {
        return std::u32string_view(m_text).substr(m_lineStarts[lineNumber - 1],
                                                  m_lineStarts[lineNumber] - m_lineStarts[lineNumber - 1] - 1);
    }

    bool hasLine(std::size_t lineNumber)
    {
        while (lineNumber >= m_lineStarts.size() && m_current != m_transcoder.end())
        {
            m_current++;
        }
        return m_lineStarts.size() > lineNumber;
    }

    tcb::span<const std::size_t> getLineStarts() const
    {
        return m_lineStarts;
    }

    std::u32string_view getText() const
    {
        return m_text;
    }

    std::string_view getFilename() const
    {
        return m_filename;
    }
};
} // namespace pylir::Diag
