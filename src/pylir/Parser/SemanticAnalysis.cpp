//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "SemanticAnalysis.hpp"

#include <llvm/ADT/ScopeExit.h>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Interfaces/Builtins.hpp>
#include <pylir/Support/ValueReset.hpp>

using namespace pylir;

void pylir::SemanticAnalysis::visit(Syntax::Yield& yield) {
  if (!m_inFunc)
    createError(yield, Diag::OCCURRENCE_OF_YIELD_OUTSIDE_OF_FUNCTION)
        .addHighlight(yield.yieldToken);

  Visitor::visit(yield);
}

void pylir::SemanticAnalysis::visit(Syntax::Atom& atom) {
  if (auto* scope = getCurrentScope();
      scope && atom.token.getTokenType() == TokenType::Identifier) {
    // emplace will only insert if it is not already contained. So it will only
    // be marked as unknown if we didn't know it's kind already
    scope->identifiers.insert(
        {IdentifierToken{atom.token}, Syntax::Scope::Kind::Unknown});
  }
}

void pylir::SemanticAnalysis::visit(Syntax::Assignment& assignment) {
  addToNamespace(assignment.variable);
  visit(*assignment.expression);
}

void pylir::SemanticAnalysis::addToNamespace(const IdentifierToken& token) {
  auto* scope = getCurrentScope();
  if (!scope) {
    m_globals->insert(token);
    return;
  }
  auto* result =
      scope->identifiers.insert({token, pylir::Syntax::Scope::Local}).first;
  if (result->second == Syntax::Scope::Unknown)
    result->second = Syntax::Scope::Local;
}

void pylir::SemanticAnalysis::addToNamespace(Syntax::Target& target) {
  class TargetVisitor : public Syntax::Visitor<TargetVisitor> {
  public:
    std::function<void(const pylir::IdentifierToken&)> callback;

    using Visitor::visit;

    void visit(Syntax::Atom& atom) {
      if (atom.token.getTokenType() == TokenType::Identifier)
        callback(IdentifierToken(atom.token));
    }

    void visit(Syntax::AttributeRef&) {}

    void visit(Syntax::Subscription&) {}

    void visit(Syntax::Slice&) {}

    void visit(Syntax::DictDisplay&) {}

    void visit(Syntax::SetDisplay&) {}

    void visit(Syntax::ListDisplay& listDisplay) {
      if (std::holds_alternative<Syntax::Comprehension>(listDisplay.variant))
        return;

      Visitor::visit(listDisplay);
    }

    void visit(Syntax::Yield&) {}

    void visit(Syntax::Generator&) {}

    void visit(Syntax::BinOp&) {}

    void visit(Syntax::Lambda&) {}

    void visit(Syntax::Call&) {}

    void visit(Syntax::UnaryOp&) {}

    void visit(Syntax::Comparison&) {}

    void visit(Syntax::Conditional&) {}

    void visit(Syntax::Assignment&) {}
  } visitor{{}, [&](const IdentifierToken& token) { addToNamespace(token); }};
  visitor.visit(target);
}

void pylir::SemanticAnalysis::visit(Syntax::Lambda& lambda) {
  for (auto& iter : lambda.parameters)
    visit(iter);

  {
    ValueReset inLoopReset(m_inLoop);
    ValueReset inFuncReset(m_inFunc);
    ValueReset currentScopeReset(m_currentScopeOwner);
    m_inLoop = false;
    m_inFunc = true;
    m_currentScopeOwner = &lambda;
    for (auto& iter : lambda.parameters)
      addToNamespace(iter.name);

    visit(*lambda.expression);
  }
  if (!m_currentScopeOwner) {
    // Lambda namespaces are only finished if in the global scope or class. If
    // nested within a function, nonlocals may still be defined below the
    // lambda, hence name resolution must happen at the end of the outermost
    // function.
    finishNamespace(&lambda);
  }
}

void pylir::SemanticAnalysis::visit(pylir::Syntax::CompFor& compFor) {
  addToNamespace(*compFor.targets);
  visit(*compFor.test);
  pylir::match(
      compFor.compIter, [](std::monostate) {},
      [&](auto& ptr) { return visit(*ptr); });
}

void pylir::SemanticAnalysis::visit(Syntax::ReturnStmt& returnStmt) {
  if (!m_inFunc)
    createError(returnStmt.returnKeyword,
                Diag::OCCURRENCE_OF_RETURN_OUTSIDE_OF_FUNCTION)
        .addHighlight(returnStmt.returnKeyword);

  Visitor::visit(returnStmt);
}

void pylir::SemanticAnalysis::visit(Syntax::SingleTokenStmt& singleTokenStmt) {
  if (!m_inLoop &&
      (singleTokenStmt.token.getTokenType() == TokenType::BreakKeyword ||
       singleTokenStmt.token.getTokenType() == TokenType::ContinueKeyword))
    createError(singleTokenStmt.token, Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP,
                singleTokenStmt.token.getTokenType())
        .addHighlight(singleTokenStmt.token);

  Visitor::visit(singleTokenStmt);
}

void pylir::SemanticAnalysis::visit(
    Syntax::GlobalOrNonLocalStmt& globalOrNonLocalStmt) {
  auto* scope = getCurrentScope();
  if (!scope) {
    if (globalOrNonLocalStmt.token.getTokenType() == TokenType::NonlocalKeyword)
      return;

    for (auto& iter : globalOrNonLocalStmt.identifiers)
      m_globals->insert(iter);

    return;
  }

  std::function<void(const IdentifierToken&)> handler;
  if (globalOrNonLocalStmt.token.getTokenType() == TokenType::NonlocalKeyword) {
    handler = [&](const IdentifierToken& nonLocal) {
      if (auto [result, inserted] =
              scope->identifiers.insert({nonLocal, Syntax::Scope::NonLocal});
          !inserted) {
        switch (result->second) {
        case Syntax::Scope::Kind::Local:
        case Syntax::Scope::Kind::Cell:
          createError(
              nonLocal,
              Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_LOCAL_VARIABLE,
              nonLocal.getValue())
              .addHighlight(nonLocal)
              .addNote(result->first, Diag::LOCAL_VARIABLE_N_BOUND_HERE,
                       nonLocal.getValue())
              .addHighlight(result->first);
          break;
        case Syntax::Scope::Kind::Global:
          createError(
              nonLocal,
              Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_GLOBAL_VARIABLE,
              nonLocal.getValue())
              .addHighlight(nonLocal)
              .addNote(result->first, Diag::GLOBAL_VARIABLE_N_BOUND_HERE,
                       nonLocal.getValue())
              .addHighlight(result->first);
          break;
        case Syntax::Scope::Kind::Unknown:
          createError(nonLocal, Diag::NONLOCAL_N_USED_PRIOR_TO_DECLARATION,
                      nonLocal.getValue())
              .addHighlight(nonLocal)
              .addNote(result->first, Diag::N_USED_HERE, nonLocal.getValue())
              .addHighlight(result->first);
          break;
        case Syntax::Scope::Kind::NonLocal: break;
        }
      }
    };
  } else {
    handler = [&](const IdentifierToken& global) {
      if (auto [result, inserted] =
              scope->identifiers.insert({global, Syntax::Scope::Global});
          !inserted) {
        switch (result->second) {
        case Syntax::Scope::Kind::Local:
        case Syntax::Scope::Kind::Cell:
          createError(
              global,
              Diag::DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_LOCAL_VARIABLE,
              global.getValue())
              .addHighlight(global)
              .addNote(result->first, Diag::LOCAL_VARIABLE_N_BOUND_HERE,
                       global.getValue())
              .addHighlight(result->first);
          break;
        case Syntax::Scope::Kind::NonLocal:
          createError(
              global,
              Diag::DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_NONLOCAL_VARIABLE,
              global.getValue())
              .addHighlight(global)
              .addNote(result->first, Diag::NONLOCAL_VARIABLE_N_BOUND_HERE,
                       global.getValue())
              .addHighlight(result->first);
          break;
        case Syntax::Scope::Kind::Unknown:
          createError(global, Diag::GLOBAL_N_USED_PRIOR_TO_DECLARATION,
                      global.getValue())
              .addHighlight(global)
              .addNote(result->first, Diag::N_USED_HERE, global.getValue())
              .addHighlight(result->first);
          break;
        case Syntax::Scope::Kind::Global: break;
        }
      }
      m_globals->insert(global);
    };
  }
  llvm::for_each(globalOrNonLocalStmt.identifiers, handler);
}

void pylir::SemanticAnalysis::visit(Syntax::AssignmentStmt& assignmentStmt) {
  for (auto& iter : assignmentStmt.targets)
    addToNamespace(*iter.first);

  if (assignmentStmt.maybeAnnotation)
    visit(*assignmentStmt.maybeAnnotation);

  if (assignmentStmt.maybeExpression)
    visit(*assignmentStmt.maybeExpression);
}

void pylir::SemanticAnalysis::visit(Syntax::DelStmt& delStmt) {
  addToNamespace(*delStmt.targetList);
}

void pylir::SemanticAnalysis::visit(Syntax::WithStmt& withStmt) {
  for (auto& iter : withStmt.items) {
    visit(*iter.expression);
    if (iter.maybeTarget)
      addToNamespace(*iter.maybeTarget);
  }
  visit(*withStmt.suite);
}

void pylir::SemanticAnalysis::visit(Syntax::WhileStmt& whileStmt) {
  visit(*whileStmt.condition);
  {
    pylir::ValueReset inLoopReset(m_inLoop);
    m_inLoop = true;
    visit(*whileStmt.suite);
  }
  if (whileStmt.elseSection)
    visit(*whileStmt.elseSection->suite);
}

void pylir::SemanticAnalysis::visit(Syntax::ForStmt& forStmt) {
  addToNamespace(*forStmt.targetList);
  visit(*forStmt.expression);
  {
    pylir::ValueReset inLoopReset(m_inLoop);
    m_inLoop = true;
    visit(*forStmt.suite);
  }
  if (forStmt.elseSection)
    visit(*forStmt.elseSection->suite);
}

void pylir::SemanticAnalysis::visit(Syntax::TryStmt& tryStmt) {
  visit(*tryStmt.suite);
  for (auto& iter : tryStmt.excepts) {
    visit(*iter.filter);
    if (iter.maybeName)
      addToNamespace(*iter.maybeName);

    visit(*iter.suite);
  }
  if (tryStmt.maybeExceptAll)
    visit(*tryStmt.maybeExceptAll->suite);

  if (tryStmt.elseSection)
    visit(*tryStmt.elseSection->suite);

  if (tryStmt.finally)
    visit(*tryStmt.finally->suite);
}

void pylir::SemanticAnalysis::finishNamespace(ScopeOwner owner) {
  class NamespaceVisitor : public pylir::Syntax::Visitor<NamespaceVisitor> {
    void finishNamespace(Syntax::Scope& scope) {
      for (auto& [id, kind] : scope.identifiers) {
        switch (kind) {
        case pylir::Syntax::Scope::NonLocal: {
          if (std::none_of(
                  m_scopes.begin(), m_scopes.end(),
                  [&id = id](const pylir::Syntax::Scope* scope) -> bool {
                    return scope->identifiers.count(id);
                  })) {
            m_analysis
                ->createError(id,
                              Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES,
                              id.getValue())
                .addHighlight(id);
            break;
          }
          break;
        }
        case pylir::Syntax::Scope::Unknown: {
          for (auto& iter : llvm::reverse(m_scopes)) {
            const auto* res = iter->identifiers.find(id);
            if (res != iter->identifiers.end()) {
              switch (res->second) {
              case pylir::Syntax::Scope::Global:
                kind = pylir::Syntax::Scope::Global;
                break;
              case pylir::Syntax::Scope::Unknown: continue;
              case pylir::Syntax::Scope::Cell:
              case pylir::Syntax::Scope::Local:
              case pylir::Syntax::Scope::NonLocal:
                kind = pylir::Syntax::Scope::NonLocal;
                break;
              }
              break;
            }
          }
          break;
        }
        default: break;
        }
      }

      if (!m_parentDef)
        return;

      // add any non locals from nested functions except if they are local to
      // this function aka the referred to local
      for (const auto& [id, kind] : scope.identifiers) {
        if (kind != pylir::Syntax::Scope::NonLocal)
          continue;
        llvm::TypeSwitch<ScopeOwner>(m_parentDef)
            .Case<pylir::Syntax::FuncDef*, pylir::Syntax::Lambda*>(
                [&id = id](auto* func) {
                  auto iter = func->scope.identifiers
                                  .insert({id, pylir::Syntax::Scope::NonLocal})
                                  .first;
                  if (iter->second == pylir::Syntax::Scope::Local) {
                    iter->second = pylir::Syntax::Scope::Cell;
                  }
                })
            .template Case<pylir::Syntax::ClassDef*>(
                [&id = id](pylir::Syntax::ClassDef* classDef) {
                  classDef->scope.identifiers.insert(
                      {id, pylir::Syntax::Scope::NonLocal});
                });
      }
    }

    std::vector<const pylir::Syntax::Scope*> m_scopes;
    ScopeOwner m_parentDef;
    SemanticAnalysis* m_analysis;

  public:
    explicit NamespaceVisitor(SemanticAnalysis& analysis)
        : m_analysis(&analysis) {}

    using Visitor::visit;

    void visit(pylir::Syntax::ClassDef& classDef) {
      {
        pylir::ValueReset reset(m_parentDef);
        m_parentDef = &classDef;
        Visitor::visit(classDef);
      }
      finishNamespace(classDef.scope);
    }

    void visit(pylir::Syntax::FuncDef& funcDef) {
      {
        m_scopes.push_back(&funcDef.scope);
        auto exit = llvm::make_scope_exit([&] { m_scopes.pop_back(); });
        pylir::ValueReset reset(m_parentDef);
        m_parentDef = &funcDef;
        Visitor::visit(funcDef);
      }
      finishNamespace(funcDef.scope);
    }

    void visit(pylir::Syntax::Lambda& lambda) {
      {
        m_scopes.push_back(&lambda.scope);
        auto exit = llvm::make_scope_exit([&] { m_scopes.pop_back(); });
        pylir::ValueReset reset(m_parentDef);
        m_parentDef = &lambda;
        Visitor::visit(lambda);
      }
      finishNamespace(lambda.scope);
    }
  } visitor{*this};
  llvm::TypeSwitch<ScopeOwner>(owner)
      .Case<Syntax::Lambda*, Syntax::FuncDef*, Syntax::ClassDef*>(
          [&](auto* ptr) { visitor.visit(*ptr); });
}

namespace {

/// Checks whether 'decorators' contains '@pylir.intr.const_export'. Returns
/// the instance of 'Syntax::Intrinsic' if contained.
Syntax::Intrinsic*
hasConstDecorator(llvm::ArrayRef<Syntax::Decorator> decorators) {
  for (const Syntax::Decorator& decorator : decorators) {
    auto* intrinsic = decorator.expression->dyn_cast<Syntax::Intrinsic>();
    if (intrinsic && intrinsic->name == "pylir.intr.const_export")
      return intrinsic;
  }
  return nullptr;
}
} // namespace

void pylir::SemanticAnalysis::verifyCommonConstDecorator(
    llvm::ArrayRef<Syntax::Decorator> decorators, BaseToken nameLocation,
    bool& isExported, bool& isConst) {
  Syntax::Intrinsic* isConstExport = hasConstDecorator(decorators);
  if (isConstExport) {
    isExported = true;
    if (m_currentScopeOwner) {
      createError(nameLocation,
                  Diag::CONST_EXPORT_OBJECT_MUST_BE_DEFINED_IN_GLOBAL_SCOPE)
          .addHighlight(nameLocation)
          .addHighlight(*isConstExport, Diag::flags::secondaryColour);
    }
  }

  isConst = isConstExport || m_inConstClass;
  if (!isConst)
    return;

  for (const Syntax::Decorator& decorator : decorators) {
    if (llvm::isa<Syntax::Intrinsic>(*decorator.expression))
      continue;

    createError(decorator,
                Diag::DECORATORS_ON_A_CONST_EXPORT_OBJECT_ARE_NOT_SUPPORTED)
        .addHighlight(decorator)
        .addHighlight(nameLocation, Diag::flags::secondaryColour);
  }
}

void pylir::SemanticAnalysis::visit(Syntax::FuncDef& funcDef) {
  verifyCommonConstDecorator(funcDef.decorators, funcDef.funcName,
                             funcDef.isExported, funcDef.isConst);

  for (auto& iter : funcDef.decorators)
    visit(iter);

  addToNamespace(funcDef.funcName);
  for (auto& iter : funcDef.parameterList) {
    visit(iter);
    if (iter.maybeDefault && funcDef.isConst)
      verifyIsConstant(*iter.maybeDefault);
  }

  if (funcDef.maybeSuffix)
    visit(*funcDef.maybeSuffix);

  {
    ValueReset inLoopReset(m_inLoop);
    ValueReset inFuncReset(m_inFunc);
    ValueReset currentScopeOwnerReset(m_currentScopeOwner);
    m_inLoop = false;
    m_inFunc = true;
    m_currentScopeOwner = &funcDef;
    for (auto& iter : funcDef.parameterList) {
      addToNamespace(iter.name);
    }
    visit(*funcDef.suite);
  }
  if (!m_currentScopeOwner) {
    // this indicates that this funcdef is at global scope or only nested in
    // classes. We now need to resolve any nonlocals inside any nested funcDefs
    // and figure out whether any unknowns are nonlocal or global

    for (auto& [iter, kind] : funcDef.scope.identifiers) {
      if (kind != Syntax::Scope::NonLocal)
        continue;
      createError(iter, Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES,
                  iter.getValue())
          .addHighlight(iter);
    }
    finishNamespace(&funcDef);
  }
}

void pylir::SemanticAnalysis::visit(Syntax::ClassDef& classDef) {
  verifyCommonConstDecorator(classDef.decorators, classDef.className,
                             classDef.isExported, classDef.isConst);

  std::optional<ValueReset<bool>> constClassReset;
  if (classDef.isConst) {
    constClassReset.emplace(m_inConstClass);
    m_inConstClass = true;

    // Verify that the body of the class is valid. Currently only assignments of
    // the form "id = constant-expr", function definitions, and "pass" are
    // valid.
    for (const Syntax::Suite::Variant& variant : classDef.suite->statements) {
      bool valid = match(
          variant,
          [&](const IntrVarPtr<Syntax::SimpleStmt>& simpleStatement) {
            return simpleStatement->match(
                [&](const Syntax::AssignmentStmt& assignmentStmt) -> bool {
                  if (assignmentStmt.targets.size() != 1)
                    return false;

                  if (assignmentStmt.targets.front().second.getTokenType() !=
                      TokenType::Assignment)
                    return false;

                  if (assignmentStmt.maybeExpression)
                    verifyIsConstant(*assignmentStmt.maybeExpression);
                  return llvm::isa<Syntax::Atom>(
                      *assignmentStmt.targets.front().first);
                },
                [&](const Syntax::SingleTokenStmt& singleTokenStmt) -> bool {
                  return singleTokenStmt.token.getTokenType() ==
                         TokenType::PassKeyword;
                },
                [](const auto&) { return false; });
          },
          [&](const IntrVarPtr<Syntax::CompoundStmt>& compoundStatement) {
            return llvm::isa<Syntax::FuncDef>(*compoundStatement);
          });
      if (valid)
        continue;

      Diag::Location loc = Diag::rangeLoc(variant);
      createError(
          loc,
          Diag::
              ONLY_SINGLE_ASSIGNMENTS_AND_FUNCTION_DEFINITIONS_ALLOWED_IN_CONST_EXPORT_CLASS)
          .addHighlight(loc)
          .addHighlight(classDef.className, Diag::flags::secondaryColour);
    }
  }

  for (auto& iter : classDef.decorators)
    visit(iter);

  addToNamespace(classDef.className);
  if (classDef.inheritance)
    for (auto& iter : classDef.inheritance->argumentList) {
      visit(iter);
      if (classDef.isConst) {
        if (iter.maybeExpansionsOrEqual) {
          createError(
              iter,
              Diag::
                  ONLY_POSITIONAL_ARGUMENTS_ALLOWED_IN_CONST_EXPORT_CLASS_INHERITANCE_LIST)
              .addHighlight(iter)
              .addHighlight(classDef.className, Diag::flags::secondaryColour);
        }
        verifyIsConstant(*iter.expression);
      }
    }

  {
    ValueReset inLoopReset(m_inLoop);
    ValueReset inFuncReset(m_inFunc);
    ValueReset currentScopeOwnerReset(m_currentScopeOwner);
    m_inLoop = false;
    m_inFunc = false;
    m_currentScopeOwner = &classDef;
    visit(*classDef.suite);
  }
  if (!m_currentScopeOwner)
    finishNamespace(&classDef);
}

namespace {

/// Visitor assuming a constant expression until proven otherwise.
class IsConstantExpressionVisitor
    : public Syntax::Visitor<IsConstantExpressionVisitor> {
  bool m_isConstant = true;

public:
  using Visitor::Visitor;

  template <class T>
  void visit(T&) {
    m_isConstant = false;
  }

  /// Top-level 'visit' function returning true if 'expression' is a constant
  /// expression.
  [[nodiscard]] bool visit(Syntax::Expression& expression) {
    Visitor::visit(expression);
    return m_isConstant;
  }

  void visit(Syntax::StarredItem& starredItem) {
    if (!starredItem.maybeStar) {
      (void)visit(*starredItem.expression);
      return;
    }
    m_isConstant = false;
  }

  void visit(Syntax::TupleConstruct& tupleConstruct) {
    // Tuple is constant if all elements are.
    Visitor::visit(tupleConstruct);
  }

  void visit(Syntax::Intrinsic&) {
    // Be very permissive with intrinsics for now.
  }

  void visit(Syntax::Atom& atom) {
    // Atoms are all constants except identifiers. We bend the rules of the
    // language here by saying "all references that area possibly to builtins
    // are builtins and constant, otherwise it's not a constant expression".
    switch (atom.token.getTokenType()) {
    case TokenType::IntegerLiteral:
    case TokenType::FloatingPointLiteral:
    case TokenType::StringLiteral:
    case TokenType::TrueKeyword:
    case TokenType::FalseKeyword:
    case TokenType::NoneKeyword:
    case TokenType::ByteLiteral:
    case TokenType::ComplexLiteral: return;
    case TokenType::Identifier:

      // Name must match a public builtin in the 'builtins' module with the
      // spelling being equal to the 'builtins.' prefix removed.
      if (llvm::none_of(Builtins::allBuiltins, [&](Builtins::Builtin builtin) {
            return builtin.isPublic && builtin.name.starts_with("builtins.") &&
                   builtin.name.drop_front(
                       std::string_view("builtins.").size()) ==
                       pylir::get<std::string>(atom.token.getValue());
          }))
        m_isConstant = false;
      return;
    default: PYLIR_UNREACHABLE;
    }
  }
};
} // namespace

void pylir::SemanticAnalysis::verifyIsConstant(Syntax::Expression& expression) {
  IsConstantExpressionVisitor visitor;
  if (visitor.visit(expression))
    return;

  createError(expression, Diag::EXPECTED_CONSTANT_EXPRESSION)
      .addHighlight(expression);
}
