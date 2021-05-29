
#pragma once

#include <pylir/Support/LazyCacheIterator.hpp>
#include <pylir/Support/Text.hpp>

#include <string>
#include <string_view>

namespace pylir::Diag
{
class Document
{
    std::string m_filename;
    Text::Encoding m_encoding;
    std::string m_input;
    std::u32string m_transcoded;
    std::vector<std::size_t> m_lineStarts{0};
    Text::Transcoder<void, char32_t> m_transcoder;

public:
    Document(std::string filename, std::string input)
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

    }
};
} // namespace pylir::Diag
