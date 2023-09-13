//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DiagnosticsManager.hpp"

#include <llvm/Support/raw_ostream.h>

#include "DiagnosticsBuilder.hpp"

pylir::Diag::DiagnosticsManager::DiagnosticsManager(
    std::function<void(Diagnostic&&)> diagnosticCallback) {
  if (diagnosticCallback) {
    m_diagnosticCallback = std::move(diagnosticCallback);
    return;
  }
  m_diagnosticCallback = [](Diagnostic&& diag) { llvm::errs() << diag; };
}

void pylir::Diag::SubDiagnosticsManagerBase::report(Diagnostic&& diag) {
  if (diag.messages.front().severity == Severity::Error) {
    m_errorsOccurred = true;
    m_parent->m_errorsOccurred = true;
  }
  std::unique_lock lock(m_parent->m_diagCallbackMutex);
  m_parent->m_diagnosticCallback(std::move(diag));
}

const pylir::Diag::Warning* pylir::Diag::SubDiagnosticsManagerBase::getWarning(
    llvm::StringRef warningName) const {
  return m_parent->getWarning(warningName);
}
