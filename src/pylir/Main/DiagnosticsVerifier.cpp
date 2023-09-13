//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DiagnosticsVerifier.hpp"

#include <llvm/ADT/StringSwitch.h>

#include <ctre.hpp>

mlir::LogicalResult pylir::DiagnosticsVerifier::checkExpected(
    const Diag::Diagnostic::Message& message,
    decltype(m_fileLineToExpectedMessages)::iterator res) {
  if (res == m_fileLineToExpectedMessages.end())
    return mlir::failure();

  auto expectedIter = llvm::find_if(res->second, [&](const Expected& expected) {
    if (expected.kind != message.severity)
      return false;

    if (!expected.regex)
      return expected.message == message.message;

    return expected.regex->match(message.message);
  });
  if (expectedIter == res->second.end())
    return mlir::failure();

  res->second.erase(expectedIter);
  if (res->second.empty())
    m_fileLineToExpectedMessages.erase(res);

  return mlir::success();
}

pylir::DiagnosticsVerifier::DiagnosticsVerifier(
    Diag::DiagnosticsManager& manager) {
  manager.setDiagnosticCallback([&](Diag::Diagnostic&& diagnostic) {
    for (auto& iter : diagnostic.messages) {
      auto* res = m_fileLineToExpectedMessages.find(
          {iter.document,
           iter.document ? iter.document->getLineNumber(iter.location) : 0});
      if (mlir::succeeded(checkExpected(iter, res)))
        continue;

      m_errorsOccurred = true;
      std::vector<Diag::Diagnostic::Message> messages;
      messages.push_back({Diag::Severity::Error,
                          nullptr,
                          0,
                          "Did not expect diagnostic:",
                          {}});
      messages.push_back(std::move(iter));
      llvm::errs() << Diag::Diagnostic{std::move(messages)};
    }
  });
}

constexpr static auto PATTERN = ctll::fixed_string{
    R"(expected-(?<kind>error|note|warning)(?<re>-re)?\h*(?:@(?:(?<rel>[+\-][0-9]+)|(?<above>above)|(?<below>below)))?(?<text>(?:\h*\{\{.*\}\})+))"};

constexpr static auto KIND = ctll::fixed_string{"kind"};
constexpr static auto REL = ctll::fixed_string{"rel"};
constexpr static auto RE = ctll::fixed_string{"re"};
constexpr static auto ABOVE = ctll::fixed_string{"above"};
constexpr static auto BELOW = ctll::fixed_string{"below"};

constexpr static auto TEXT = ctll::fixed_string{"text"};
constexpr static auto TEXT_WITHIN_PATTERN =
    ctll::fixed_string{R"(\{\{(?<text>.*)\}\})"};

void pylir::DiagnosticsVerifier::addDocument(
    const pylir::Diag::Document& document) {
  for (const auto& iter : ctre::multiline_range<PATTERN>(document.getText())) {
    Diag::Severity severity;
    if (iter.get<KIND>() == U"error")
      severity = Diag::Severity::Error;
    else if (iter.get<KIND>() == U"note")
      severity = Diag::Severity::Note;
    else if (iter.get<KIND>() == U"warning")
      severity = Diag::Severity::Warning;

    std::size_t line = document.getLineNumber(iter.data() - document.begin());
    if (iter.get<REL>()) {
      auto view = iter.get<REL>().view();
      bool add = view.front() == U'+';
      view = view.substr(1);
      std::size_t offset = 0;
      for (char32_t c : view)
        offset = offset * 10 + (c - U'0');

      if (add)
        line += offset;
      else
        line -= offset;

    } else if (iter.get<ABOVE>()) {
      line--;
    } else if (iter.get<BELOW>()) {
      line++;
    }
    for (const auto& iter2 :
         ctre::multiline_range<TEXT_WITHIN_PATTERN>(iter.get<TEXT>())) {
      auto textWithin = iter2.get<TEXT>();
      std::size_t start = textWithin.data() - document.begin();
      std::size_t end = start + textWithin.size();
      if (!iter.get<RE>()) {
        m_fileLineToExpectedMessages[{&document, line}].push_back(
            {start, end, severity, nullptr,
             Text::toUTF8String(textWithin.view())});
        continue;
      }
      std::string regexRes;
      llvm::raw_string_ostream regexOS(regexRes);
      std::u32string_view strToProcess = textWithin.view();
      while (!strToProcess.empty()) {
        // Find the next regex block.
        size_t regexIt = strToProcess.find(U"{{");
        if (regexIt == std::string::npos) {
          regexOS << llvm::Regex::escape(Text::toUTF8String(strToProcess));
          break;
        }
        regexOS << llvm::Regex::escape(
            Text::toUTF8String(strToProcess.substr(0, regexIt)));
        strToProcess = strToProcess.substr(regexIt + 2);

        // Find the end of the regex block.
        size_t regexEndIt = strToProcess.find(U"}}");
        if (regexEndIt == std::string::npos) {
          m_errorsOccurred = true;
          auto openBracketPos =
              static_cast<std::size_t>(strToProcess.data() - document.begin()) -
              2;
          llvm::errs() << Diag::DiagnosticsBuilder(
                              document, Diag::Severity::Error, openBracketPos,
                              "found start of regex with no end '}}}}'")
                              .addHighlight(openBracketPos, openBracketPos + 1)
                              .getDiagnostic();
          continue;
        }
        std::string regexStr =
            Text::toUTF8String(strToProcess.substr(0, regexEndIt));

        // Validate that the regex is actually valid.
        std::string regexError;
        if (!llvm::Regex(regexStr).isValid(regexError)) {
          m_errorsOccurred = true;
          std::size_t regexStart = strToProcess.data() - document.begin();
          std::size_t regexEnd = regexStart + regexEndIt;
          llvm::errs() << Diag::DiagnosticsBuilder(
                              document, Diag::Severity::Error, regexStart,
                              "invalid regex '{}'", regexStr)
                              .addHighlight(regexStart, regexEnd - 1,
                                            std::move(regexError))
                              .getDiagnostic();
          continue;
        }

        regexOS << '(' << regexStr << ')';
        strToProcess = strToProcess.substr(regexEndIt + 2);
      }
      m_fileLineToExpectedMessages[{&document, line}].push_back(
          {start, end, severity,
           std::make_unique<llvm::Regex>(std::move(regexRes)), ""});
    }
  }
}

mlir::LogicalResult pylir::DiagnosticsVerifier::verify() {
  for (auto& iter : m_fileLineToExpectedMessages) {
    m_errorsOccurred = true;
    for (auto& iter2 : iter.second) {
      std::string_view strRep;
      switch (iter2.kind) {
      case Diag::Severity::Error: strRep = "error"; break;
      case Diag::Severity::Note: strRep = "note"; break;
      case Diag::Severity::Warning: strRep = "warning"; break;
      }
      llvm::errs() << Diag::DiagnosticsBuilder(
                          *iter.first.first, Diag::Severity::Error, iter2.start,
                          "Did not encounter {} at line {}:", strRep,
                          iter.first.second)
                          .addHighlight(iter2.start, iter2.end - 1)
                          .getDiagnostic();
    }
  }
  return mlir::failure(m_errorsOccurred);
}
