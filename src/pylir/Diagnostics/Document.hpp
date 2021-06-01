
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

public:
    using value_type = char32_t;
    using reference = const char32_t&;
    using const_reference = const char32_t&;
    using iterator = const char32_t*;
    using const_iterator = iterator;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    Document(std::string input, std::string filename = "<stdin>", Text::Encoding encoding = Text::Encoding::UTF8)
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

    [[nodiscard]] iterator begin() const
    {
        return m_text.data();
    }

    [[nodiscard]] const_iterator cbegin() const
    {
        return begin();
    }

    [[nodiscard]] iterator end() const
    {
        return m_text.data() + m_text.size();
    }

    [[nodiscard]] const_iterator cend() const
    {
        return end();
    }

    [[nodiscard]] std::size_t getLineNumber(std::size_t offset) const
    {
        auto result = std::lower_bound(m_lineStarts.begin(), m_lineStarts.end(), offset);
        if (result != m_lineStarts.end() && *result == offset)
        {
            result++;
        }
        return result - m_lineStarts.begin();
    }

    [[nodiscard]] std::size_t getColNumber(std::size_t offset) const
    {
        auto lineNumber = getLineNumber(offset);
        return offset - m_lineStarts[lineNumber - 1] + 1;
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> getLineCol(std::size_t offset) const
    {
        auto lineNumber = getLineNumber(offset);
        return {lineNumber, offset - m_lineStarts[lineNumber - 1] + 1};
    }

    [[nodiscard]] std::u32string_view getLine(std::size_t lineNumber) const
    {
        return std::u32string_view(m_text).substr(m_lineStarts[lineNumber - 1],
                                                  m_lineStarts[lineNumber] - m_lineStarts[lineNumber - 1] - 1);
    }

    [[nodiscard]] bool hasLine(std::size_t lineNumber)
    {
        return m_lineStarts.size() > lineNumber;
    }

    [[nodiscard]] tcb::span<const std::size_t> getLineStarts() const
    {
        return m_lineStarts;
    }

    [[nodiscard]] std::u32string_view getText() const
    {
        return m_text;
    }

    [[nodiscard]] std::string_view getFilename() const
    {
        return m_filename;
    }
};
} // namespace pylir::Diag
