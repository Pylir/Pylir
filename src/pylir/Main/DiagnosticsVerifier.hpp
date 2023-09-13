//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/Support/Regex.h>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Diagnostics/DiagnosticsManager.hpp>

#include <memory>

namespace pylir {
/// Class that may be attached to a 'DiagnosticsManager' to verify that the
/// compiler exactly produces the expected diagnostics. The current syntax and
/// capabilities are taken from mlir-opt. clangs implementation is even more
/// capable and may be implemented in the future.
///
/// Current syntax works by matching the following regex in any documents added:
/// expected-(error|note|warning)(-re)? (@([+\-][0-9]+)|above|below))? ({{.*}})+
///
/// (for exposition only, horizontal whitespace characters inbetween groups are
/// ignored).
///
/// The first group designates whether an error, note or warning is expected.
/// The optional (-re) suffix designates whether the pattern at the very end may
/// contain regular expressions or not. These have to then be placed inbetween
/// '{{' and '}}' strings. If this is not the case, the error messages have to
/// match the text exactly.
///
/// By default, only diagnostics emitted on the same line as the occurrence of
/// the 'expected' comment are matched. One may optionally also add a relative
/// offset via the group started with '@'. This may either be a relative offset
/// consisting of + or - followed by an integer, or the constants 'above' and
/// 'below'. The former applies the given offset to the line number of
/// 'expected' as the line number where the diagnostic should be expected.
/// '@above' and '@below' are equivalent to '@-1' and '@+1' respectively.
///
/// Example:
/// # expected-error {{an error}}
///
/// This expects that an error is emitted on the exact same line as 'expected'
/// and that its message is exactly equal to "an error".
///
/// Any diagnostics emitted during complication are matched according to the
/// above criteria (that is: category, expected line number and message text or
/// regex). If any diagnostics appear that do not have a corresponding
/// 'expected' line, it is considered an error. Equivalently, if an 'expected'
/// line exists, and no corresponding diagnostic is emitted, is is also
/// considered an error.
class DiagnosticsVerifier {
  struct Expected {
    std::size_t start;
    std::size_t end;
    Diag::Severity kind;
    std::unique_ptr<llvm::Regex> regex;
    std::string message;
  };

  bool m_errorsOccurred = false;
  llvm::MapVector<std::pair<const Diag::Document*, std::size_t>,
                  std::vector<Expected>>
      m_fileLineToExpectedMessages;

  mlir::LogicalResult
  checkExpected(const Diag::Diagnostic::Message& message,
                decltype(m_fileLineToExpectedMessages)::iterator res);

public:
  /// Installs this verifier onto the given 'DiagnosticManager', intercepting
  /// its diagnostics and matching them against any 'expected' lines found.
  explicit DiagnosticsVerifier(Diag::DiagnosticsManager& manager);

  /// Adds a document that may contain 'expected' lines to match diagnostics
  /// against.
  void addDocument(const Diag::Document& document);

  /// Does final verifications if all 'expected' lines had a corresponding
  /// diagnostic. Returns failure if any errors ever occurred.
  mlir::LogicalResult verify();
};
} // namespace pylir
