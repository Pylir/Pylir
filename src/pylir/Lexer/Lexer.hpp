
#pragma once

#include <pylir/Support/Macros.hpp>
#include <pylir/Support/Text.hpp>

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "Token.hpp"

namespace pylir
{
class Lexer
{
    class Iterator
    {
        Lexer* m_lexer;
        std::size_t m_index;

    public:
        using difference_type = std::ptrdiff_t;
        using value_type = Token;
        using pointer = const Token*;
        using reference = const Token&;
        using iterator_category = std::forward_iterator_tag;

        Iterator() = default;

        Iterator(Lexer& lexer, std::size_t index) : m_lexer(&lexer), m_index(index) {}

        reference operator*() const
        {
            PYLIR_ASSERT(m_lexer);
            PYLIR_ASSERT(m_index < m_lexer->m_tokens.size());
            return m_lexer->m_tokens[m_index];
        }

        Iterator& operator++()
        {
            PYLIR_ASSERT(m_lexer);
            if (m_index + 1 >= m_lexer->m_tokens.size())
            {
                if (!m_lexer->parseNext())
                {
                    m_index = -1;
                    return *this;
                }
            }
            m_index++;
            return *this;
        }

        Iterator operator++(int)
        {
            auto copy = *this;
            operator++();
            return copy;
        }

        difference_type operator-(const Iterator& rhs) const
        {
            return m_index - rhs.m_index;
        }

        bool operator==(const Iterator& rhs) const
        {
            return m_lexer == rhs.m_lexer && m_index == rhs.m_index;
        }

        bool operator!=(const Iterator& rhs) const
        {
            return !(rhs == *this);
        }

        pointer operator->() const
        {
            return &operator*();
        }

        friend void swap(Iterator& lhs, Iterator& rhs)
        {
            std::swap(lhs.m_lexer, rhs.m_lexer);
            std::swap(lhs.m_index, rhs.m_index);
        }
    };

    friend class Iterator;

    int m_fileId;
    std::string_view m_source;
    std::vector<Token> m_tokens;
    std::vector<int> m_lineStarts{0};
    bool pastFirstTwoLines = false;
    Text::Encoding m_encoding;
    std::optional<Text::Transcoder<void, char32_t>> m_transcoder; // Not really optional, but have to late init :/
    Text::Transcoder<void, char32_t>::iterator m_current;

    bool parseNext();

public:
    using value_type = const Token;
    using reference = const Token&;
    using const_reference = reference;
    using iterator = Iterator;
    using const_iterator = iterator;
    using difference_type = Iterator::difference_type;
    using size_type = std::size_t;

    explicit Lexer(std::string_view source, int fileId);

    Lexer(const Lexer&) = delete;
    Lexer& operator=(const Lexer&) = delete;

    Lexer(Lexer&&) noexcept = default;
    Lexer& operator=(Lexer&&) noexcept = default;

    iterator begin()
    {
        if (m_tokens.empty())
        {
            if (!parseNext())
            {
                return end();
            }
        }
        return Iterator(*this, 0);
    }

    const_iterator cbegin()
    {
        return begin();
    }

    iterator end()
    {
        return Iterator(*this, static_cast<std::size_t>(-1));
    }

    const_iterator cend()
    {
        return end();
    }
};
} // namespace pylir
