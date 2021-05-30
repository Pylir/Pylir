
#pragma once

#include <pylir/Support/LazyCacheIterator.hpp>
#include <pylir/Support/Text.hpp>

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
                m_lineStarts.push_back(m_text.size());
                m_text += '\n';
                std::advance(m_current, increment);
                return true;
            }
            case '\n': m_lineStarts.push_back(m_text.size()); [[fallthrough]];
            default: m_text += *m_current++;
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

    Document(std::string input, std::string filename = "<stdin>")
        : m_filename(std::move(filename)),
          m_encoding(Text::checkForBOM(input).value_or(Text::Encoding::UTF8)),
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

    tcb::span<const std::size_t> getLineStarts() const
    {
        return m_lineStarts;
    }

    std::u32string_view getText() const
    {
        return m_text;
    }
};
} // namespace pylir::Diag
