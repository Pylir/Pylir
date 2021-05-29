#pragma once

#include <string_view>

#include <tcb/span.hpp>

namespace pylir::Diag
{
class DiagnosticsBuilder
{
    std::u32string_view m_source;
    tcb::span<int> m_lineMapping;
    std::size_t m_location;
    std::string_view m_message;

public:
    DiagnosticsBuilder(std::u32string_view source, tcb::span<int> lineMapping, std::size_t location,
                       std::string_view message)
        : m_source(source), m_lineMapping(lineMapping), m_location(location), m_message(message)
    {
    }


};

} // namespace pylir::Diag
