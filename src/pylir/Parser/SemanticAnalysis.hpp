//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/PointerUnion.h>
#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Diagnostics/DiagnosticsManager.hpp>

#include "Visitor.hpp"

namespace pylir {
class SemanticAnalysis : public Syntax::Visitor<SemanticAnalysis> {
  Diag::DiagnosticsDocManager<>* m_manager;
  IdentifierSet* m_globals = nullptr;

  using ScopeOwner =
      llvm::PointerUnion<Syntax::FuncDef*, Syntax::ClassDef*, Syntax::Lambda*>;
  ScopeOwner m_currentScopeOwner;
  bool m_inFunc = false;
  bool m_inLoop = false;
  bool m_inConstClass = false;

  [[nodiscard]] Syntax::Scope* getCurrentScope() const {
    if (!m_currentScopeOwner)
      return nullptr;

    return llvm::TypeSwitch<decltype(m_currentScopeOwner), Syntax::Scope*>(
               m_currentScopeOwner)
        .Case<Syntax::FuncDef*, Syntax::ClassDef*, Syntax::Lambda*>(
            [](auto* ptr) -> Syntax::Scope* { return &ptr->scope; });
  }

  void addToNamespace(const Token& token) {
    addToNamespace(IdentifierToken{token});
  }

  void addToNamespace(const IdentifierToken& token);

  void addToNamespace(pylir::Syntax::Target& target);

  void finishNamespace(ScopeOwner owner);

  /// Checks whether 'decorators' contains intrinsics such as 'const_export'
  /// and verifies extra constraints given by these intrinsics. 'nameLocation'
  /// is used for any diagnostic and 'isExported' and 'isConst' are set
  /// accordingly.
  void verifyCommonConstDecorator(llvm::ArrayRef<Syntax::Decorator> decorators,
                                  BaseToken nameLocation, bool& isExported,
                                  bool& isConst);

  /// Checks whether 'expression' is a constant expression, issuing a diagnostic
  /// if not.
  void verifyIsConstant(Syntax::Expression& expression);

public:
  explicit SemanticAnalysis(Diag::DiagnosticsDocManager<>& manager)
      : m_manager(&manager) {}

  using Visitor::visit;

  template <class T, class S, class... Args>
  [[nodiscard]] auto createError(const T& location, const S& message,
                                 Args&&... args) const {
    return Diag::DiagnosticsBuilder(*m_manager, Diag::Severity::Error, location,
                                    message, std::forward<Args>(args)...);
  }

  void visit(Syntax::Yield& yield);

  void visit(Syntax::Atom& atom);

  void visit(Syntax::Assignment& assignment);

  void visit(Syntax::Lambda& lambda);

  void visit(Syntax::CompFor& compFor);

  void visit(Syntax::ReturnStmt& returnStmt);

  void visit(Syntax::SingleTokenStmt& singleTokenStmt);

  void visit(Syntax::GlobalOrNonLocalStmt& globalOrNonLocalStmt);

  void visit(Syntax::AssignmentStmt& assignmentStmt);

  void visit(Syntax::DelStmt& delStmt);

  void visit(Syntax::WithStmt& withStmt);

  void visit(Syntax::WhileStmt& whileStmt);

  void visit(Syntax::ForStmt& forStmt);

  void visit(Syntax::TryStmt& tryStmt);

  void visit(Syntax::FuncDef& funcDef);

  void visit(Syntax::ClassDef& classDef);

  void visit(Syntax::FileInput& fileInput) {
    m_globals = &fileInput.globals;
    Visitor::visit(fileInput);
  }
};
} // namespace pylir
