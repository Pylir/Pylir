//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/StringMap.h>

#include <atomic>
#include <functional>
#include <mutex>

#include "Document.hpp"

namespace pylir::Diag {

class DiagnosticsManager;

struct Diagnostic;

/// Struct returned by `getWarning` of the various kinds of diagnostics
/// managers.
struct Warning {
  /// Whether the warning is enabled.
  bool enabled;
  /// Whether the warning should be treated as error.
  bool isError;
};

/// Base class of the sub diagnostics managers. Sub diagnostics managers can be
/// created by 'DiagnosticsManager's 'createSubDiagnosticManager'. Their purpose
/// is to contain and handle data that is specific for some kind of scope, that
/// scope usually being for the compilation of a single source file. That way it
/// is possible to query whether the compilation of a specific source file had
/// errors occur. Errors reported by compilation of a source file therefore
/// first go through a corresponding 'SubDiagnosticsManagerBase' subclass before
/// reaching the main 'DiagnosticsManager'.
class SubDiagnosticsManagerBase {
  bool m_errorsOccurred = false;
  DiagnosticsManager* m_parent;

  template <class T>
  friend class DiagnosticsBuilder;

  void report(Diagnostic&& diag);

protected:
  explicit SubDiagnosticsManagerBase(DiagnosticsManager* parent)
      : m_parent(parent) {}

public:
  /// Returns true if errors have been reported through this.
  [[nodiscard]] bool errorsOccurred() const {
    return m_errorsOccurred;
  }

  /// Returns Warning struct for a specific warning, or nullptr if it does not
  /// exist.
  [[nodiscard]] const Warning* getWarning(llvm::StringRef warningName) const;
};

namespace detail {
template <class Context>
class DiagnosticsDocManagerBase : public SubDiagnosticsManagerBase {
protected:
  const Document& m_document;
  const Context& m_context;

public:
  DiagnosticsDocManagerBase(const Document& doc, const Context& context,
                            DiagnosticsManager* parent)
      : SubDiagnosticsManagerBase(parent), m_document(doc), m_context(context) {
  }

  [[nodiscard]] const Context& getContext() const {
    return m_context;
  }

  [[nodiscard]] const Document& getDocument() const {
    return m_document;
  }
};

template <>
class DiagnosticsDocManagerBase<void> : public SubDiagnosticsManagerBase {
protected:
  const Document& m_document;

public:
  DiagnosticsDocManagerBase(const Document& doc, DiagnosticsManager* parent)
      : SubDiagnosticsManagerBase(parent), m_document(doc) {}

  [[nodiscard]] const Document& getDocument() const {
    return m_document;
  }
};
} // namespace detail

/// Subclass of 'SubDiagnosticsManagerBase' which references a document and
/// optionally a context. When used with 'DiagnosticsBuilder' it enables and
/// requires the use of locations to print out the offending part of source code
/// within the document.
template <class Context = void>
class DiagnosticsDocManager final
    : public detail::DiagnosticsDocManagerBase<Context> {
public:
  using detail::DiagnosticsDocManagerBase<Context>::DiagnosticsDocManagerBase;
};

/// Subclass of 'SubDiagnosticsManagerBase' for diagnostics without any file
/// location. This can be used for environmental errors or similar for example.
/// When used with 'DiagnosticsBuilder', locations can therefore not be
/// specified, and calls to 'addHighlight' are not possible.
class DiagnosticsNoDocManager final : public SubDiagnosticsManagerBase {
public:
  explicit DiagnosticsNoDocManager(DiagnosticsManager* parent)
      : SubDiagnosticsManagerBase(parent) {}
};

/// The main global diagnostics manager. It allows customization around the
/// behaviour of diagnostics, including configuring whether warnings are errors
/// or enabled as well as a callback how to handle diagnostics.
class DiagnosticsManager {
  friend class SubDiagnosticsManagerBase;

  llvm::StringMap<Warning> m_warnings;

  std::mutex m_diagCallbackMutex;
  std::function<void(Diagnostic&&)> m_diagnosticCallback;
  std::atomic_bool m_errorsOccurred = false;

public:
  /// Constructs a diagnostic manager, optionally with a callback for how to
  /// handle diagnostics. If no such callback is provided, diagnostics are
  /// simply streamed to llvm::errs() by default. Callbacks are executed under
  /// the lock of a mutex and hence do not need to take care of being threadsafe
  /// themselves.
  explicit DiagnosticsManager(std::function<void(Diagnostic&&)> diag = {});

  /// Sets the callback for how to handle diagnostics.
  /// Callbacks are executed under the lock of a mutex and hence do not need to
  /// take care of being threadsafe themselves.
  void
  setDiagnosticCallback(std::function<void(Diagnostic&&)> diagnosticCallback) {
    m_diagnosticCallback = std::move(diagnosticCallback);
  }

  /// Returns Warning struct for a specific warning, or nullptr if it does not
  /// exist.
  Warning* getWarning(llvm::StringRef warningName) {
    auto iter = m_warnings.find(warningName);
    if (iter == m_warnings.end())
      return nullptr;

    return &iter->second;
  }

  /// Returns Warning struct for a specific warning, or nullptr if it does not
  /// exist.
  const Warning* getWarning(llvm::StringRef warningName) const {
    return const_cast<DiagnosticsManager&>(*this).getWarning(warningName);
  }

  /// Creates a new sub diagnostics manager for a document.
  DiagnosticsDocManager<> createSubDiagnosticManager(const Document& document) {
    return {document, this};
  }

  /// Creates a new sub diagnostics manager for a document and optionally a
  /// context.
  template <class Context>
  DiagnosticsDocManager<Context>
  createSubDiagnosticManager(const Document& document, const Context& context) {
    return {document, context, this};
  }

  /// Creates a new sub diagnostics manager that does not have any locations.
  DiagnosticsNoDocManager createSubDiagnosticManager() {
    return DiagnosticsNoDocManager(this);
  }
};

} // namespace pylir::Diag
