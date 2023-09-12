//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Document.hpp"

pylir::Diag::Document::Document(std::string_view input, std::string filename,
                                pylir::Text::Encoding encoding)
    : m_filename(std::move(filename)) {
  m_encoding = Text::readBOM(input).value_or(encoding);
  m_text.reserve(input.size());
  auto transcoder = Text::Transcoder<void, char32_t>(input, m_encoding);
  for (auto iter = transcoder.begin(); iter != transcoder.end();) {
    switch (*iter) {
    case '\r': {
      std::size_t increment = 1;
      if (std::next(iter) != transcoder.end() && *std::next(iter) == '\n') {
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
