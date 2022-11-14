//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "SemanticAnalysis.hpp"

#include <llvm/ADT/ScopeExit.h>

#include <pylir/Diagnostics/DiagnosticMessages.hpp>
#include <pylir/Support/ValueReset.hpp>

void pylir::SemanticAnalysis::visit(Syntax::Yield& yield)
{
    if (!m_inFunc)
    {
        createError(yield, Diag::OCCURRENCE_OF_YIELD_OUTSIDE_OF_FUNCTION).addHighlight(yield.yieldToken);
    }
    Visitor::visit(yield);
}

void pylir::SemanticAnalysis::visit(Syntax::Atom& atom)
{
    if (auto* scope = getCurrentScope(); scope && atom.token.getTokenType() == TokenType::Identifier)
    {
        // emplace will only insert if it is not already contained. So it will only be marked as unknown
        // if we didn't know it's kind already
        scope->identifiers.insert({IdentifierToken{atom.token}, Syntax::Scope::Kind::Unknown});
    }
}

void pylir::SemanticAnalysis::visit(Syntax::Assignment& assignment)
{
    addToNamespace(assignment.variable);
    visit(*assignment.expression);
}

void pylir::SemanticAnalysis::addToNamespace(const IdentifierToken& token)
{
    auto* scope = getCurrentScope();
    if (!scope)
    {
        m_globals->insert(token);
        return;
    }
    auto result = scope->identifiers.insert({token, pylir::Syntax::Scope::Local}).first;
    if (result->second == Syntax::Scope::Unknown)
    {
        result->second = Syntax::Scope::Local;
    }
}

void pylir::SemanticAnalysis::addToNamespace(Syntax::Target& target)
{
    class TargetVisitor : public Syntax::Visitor<TargetVisitor>
    {
    public:
        std::function<void(const pylir::IdentifierToken&)> callback;

        using Visitor::visit;

        void visit(Syntax::Atom& atom)
        {
            if (atom.token.getTokenType() == TokenType::Identifier)
            {
                callback(IdentifierToken(atom.token));
            }
        }

        void visit(Syntax::AttributeRef&) {}

        void visit(Syntax::Subscription&) {}

        void visit(Syntax::Slice&) {}

        void visit(Syntax::DictDisplay&) {}

        void visit(Syntax::SetDisplay&) {}

        void visit(Syntax::ListDisplay& listDisplay)
        {
            if (std::holds_alternative<Syntax::Comprehension>(listDisplay.variant))
            {
                return;
            }
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

void pylir::SemanticAnalysis::visit(Syntax::Lambda& lambda)
{
    for (auto& iter : lambda.parameters)
    {
        visit(iter);
    }

    {
        ValueReset inLoopReset(m_inLoop);
        ValueReset inFuncReset(m_inFunc);
        ValueReset currentScopeReset(m_currentScopeOwner);
        m_inLoop = false;
        m_inFunc = true;
        m_currentScopeOwner = &lambda;
        for (auto& iter : lambda.parameters)
        {
            addToNamespace(iter.name);
        }
        visit(*lambda.expression);
    }
    if (!m_currentScopeOwner)
    {
        // Lambda namespaces are only finished if in the global scope or class. If nested within a function,
        // nonlocals may still be defined below the lambda, hence name resolution must happen at the end of the
        // outermost function.
        finishNamespace(&lambda);
    }
}

void pylir::SemanticAnalysis::visit(pylir::Syntax::CompFor& compFor)
{
    addToNamespace(*compFor.targets);
    visit(*compFor.test);
    pylir::match(
        compFor.compIter, [](std::monostate) {}, [&](auto& ptr) { return visit(*ptr); });
}

void pylir::SemanticAnalysis::visit(Syntax::ReturnStmt& returnStmt)
{
    if (!m_inFunc)
    {
        createError(returnStmt.returnKeyword, Diag::OCCURRENCE_OF_RETURN_OUTSIDE_OF_FUNCTION)
            .addHighlight(returnStmt.returnKeyword);
    }
    Visitor::visit(returnStmt);
}

void pylir::SemanticAnalysis::visit(Syntax::SingleTokenStmt& singleTokenStmt)
{
    if (!m_inLoop
        && (singleTokenStmt.token.getTokenType() == TokenType::BreakKeyword
            || singleTokenStmt.token.getTokenType() == TokenType::ContinueKeyword))
    {
        createError(singleTokenStmt.token, Diag::OCCURRENCE_OF_N_OUTSIDE_OF_LOOP, singleTokenStmt.token.getTokenType())
            .addHighlight(singleTokenStmt.token);
    }
    Visitor::visit(singleTokenStmt);
}

void pylir::SemanticAnalysis::visit(Syntax::GlobalOrNonLocalStmt& globalOrNonLocalStmt)
{
    auto* scope = getCurrentScope();
    if (!scope)
    {
        if (globalOrNonLocalStmt.token.getTokenType() == TokenType::NonlocalKeyword)
        {
            return;
        }
        for (auto& iter : globalOrNonLocalStmt.identifiers)
        {
            m_globals->insert(iter);
        }
        return;
    }

    std::function<void(const IdentifierToken&)> handler;
    if (globalOrNonLocalStmt.token.getTokenType() == TokenType::NonlocalKeyword)
    {
        handler = [&](const IdentifierToken& nonLocal)
        {
            if (auto [result, inserted] = scope->identifiers.insert({nonLocal, Syntax::Scope::NonLocal}); !inserted)
            {
                switch (result->second)
                {
                    case Syntax::Scope::Kind::Local:
                    case Syntax::Scope::Kind::Cell:
                        createError(nonLocal, Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_LOCAL_VARIABLE,
                                    nonLocal.getValue())
                            .addHighlight(nonLocal)
                            .addNote(result->first, Diag::LOCAL_VARIABLE_N_BOUND_HERE, nonLocal.getValue())
                            .addHighlight(result->first);
                        break;
                    case Syntax::Scope::Kind::Global:
                        createError(nonLocal, Diag::DECLARATION_OF_NONLOCAL_N_CONFLICTS_WITH_GLOBAL_VARIABLE,
                                    nonLocal.getValue())
                            .addHighlight(nonLocal)
                            .addNote(result->first, Diag::GLOBAL_VARIABLE_N_BOUND_HERE, nonLocal.getValue())
                            .addHighlight(result->first);
                        break;
                    case Syntax::Scope::Kind::Unknown:
                        createError(nonLocal, Diag::NONLOCAL_N_USED_PRIOR_TO_DECLARATION, nonLocal.getValue())
                            .addHighlight(nonLocal)
                            .addNote(result->first, Diag::N_USED_HERE, nonLocal.getValue())
                            .addHighlight(result->first);
                        break;
                    case Syntax::Scope::Kind::NonLocal: break;
                }
            }
        };
    }
    else
    {
        handler = [&](const IdentifierToken& global)
        {
            if (auto [result, inserted] = scope->identifiers.insert({global, Syntax::Scope::Global}); !inserted)
            {
                switch (result->second)
                {
                    case Syntax::Scope::Kind::Local:
                    case Syntax::Scope::Kind::Cell:
                        createError(global, Diag::DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_LOCAL_VARIABLE,
                                    global.getValue())
                            .addHighlight(global)
                            .addNote(result->first, Diag::LOCAL_VARIABLE_N_BOUND_HERE, global.getValue())
                            .addHighlight(result->first);
                        break;
                    case Syntax::Scope::Kind::NonLocal:
                        createError(global, Diag::DECLARATION_OF_GLOBAL_N_CONFLICTS_WITH_NONLOCAL_VARIABLE,
                                    global.getValue())
                            .addHighlight(global)
                            .addNote(result->first, Diag::NONLOCAL_VARIABLE_N_BOUND_HERE, global.getValue())
                            .addHighlight(result->first);
                        break;
                    case Syntax::Scope::Kind::Unknown:
                        createError(global, Diag::GLOBAL_N_USED_PRIOR_TO_DECLARATION, global.getValue())
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

void pylir::SemanticAnalysis::visit(Syntax::AssignmentStmt& assignmentStmt)
{
    for (auto& iter : assignmentStmt.targets)
    {
        addToNamespace(*iter.first);
    }
    if (assignmentStmt.maybeAnnotation)
    {
        visit(*assignmentStmt.maybeAnnotation);
    }
    if (assignmentStmt.maybeExpression)
    {
        visit(*assignmentStmt.maybeExpression);
    }
}

void pylir::SemanticAnalysis::visit(Syntax::DelStmt& delStmt)
{
    addToNamespace(*delStmt.targetList);
}

void pylir::SemanticAnalysis::visit(Syntax::WithStmt& withStmt)
{
    for (auto& iter : withStmt.items)
    {
        visit(*iter.expression);
        if (iter.maybeTarget)
        {
            addToNamespace(*iter.maybeTarget);
        }
    }
    visit(*withStmt.suite);
}

void pylir::SemanticAnalysis::visit(Syntax::WhileStmt& whileStmt)
{
    visit(*whileStmt.condition);
    {
        pylir::ValueReset inLoopReset(m_inLoop);
        m_inLoop = true;
        visit(*whileStmt.suite);
    }
    if (whileStmt.elseSection)
    {
        visit(*whileStmt.elseSection->suite);
    }
}

void pylir::SemanticAnalysis::visit(Syntax::ForStmt& forStmt)
{
    addToNamespace(*forStmt.targetList);
    visit(*forStmt.expression);
    {
        pylir::ValueReset inLoopReset(m_inLoop);
        m_inLoop = true;
        visit(*forStmt.suite);
    }
    if (forStmt.elseSection)
    {
        visit(*forStmt.elseSection->suite);
    }
}

void pylir::SemanticAnalysis::visit(Syntax::TryStmt& tryStmt)
{
    visit(*tryStmt.suite);
    for (auto& iter : tryStmt.excepts)
    {
        visit(*iter.filter);
        if (iter.maybeName)
        {
            addToNamespace(*iter.maybeName);
        }
        visit(*iter.suite);
    }
    if (tryStmt.maybeExceptAll)
    {
        visit(*tryStmt.maybeExceptAll->suite);
    }
    if (tryStmt.elseSection)
    {
        visit(*tryStmt.elseSection->suite);
    }
    if (tryStmt.finally)
    {
        visit(*tryStmt.finally->suite);
    }
}

void pylir::SemanticAnalysis::finishNamespace(ScopeOwner owner)
{
    class NamespaceVisitor : public pylir::Syntax::Visitor<NamespaceVisitor>
    {
        void finishNamespace(Syntax::Scope& scope)
        {
            for (auto& [id, kind] : scope.identifiers)
            {
                switch (kind)
                {
                    case pylir::Syntax::Scope::NonLocal:
                    {
                        if (std::none_of(m_scopes.begin(), m_scopes.end(),
                                         [&id = id](const pylir::Syntax::Scope* scope) -> bool
                                         { return scope->identifiers.count(id); }))
                        {
                            m_analysis->createError(id, Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, id.getValue())
                                .addHighlight(id);
                            break;
                        }
                        break;
                    }
                    case pylir::Syntax::Scope::Unknown:
                    {
                        for (auto& iter : llvm::reverse(m_scopes))
                        {
                            auto res = iter->identifiers.find(id);
                            if (res != iter->identifiers.end())
                            {
                                switch (res->second)
                                {
                                    case pylir::Syntax::Scope::Global: kind = pylir::Syntax::Scope::Global; break;
                                    case pylir::Syntax::Scope::Unknown: continue;
                                    case pylir::Syntax::Scope::Cell:
                                    case pylir::Syntax::Scope::Local:
                                    case pylir::Syntax::Scope::NonLocal: kind = pylir::Syntax::Scope::NonLocal; break;
                                }
                                break;
                            }
                        }
                        break;
                    }
                    default: break;
                }
            }

            // add any non locals from nested functions except if they are local to this function aka the referred
            // to local
            for (const auto& [id, kind] : scope.identifiers)
            {
                if (kind != pylir::Syntax::Scope::NonLocal)
                {
                    continue;
                }
                llvm::TypeSwitch<ScopeOwner>(m_parentDef)
                    .Case<pylir::Syntax::FuncDef*, pylir::Syntax::Lambda*>(
                        [&id = id](auto* func)
                        {
                            auto iter = func->scope.identifiers.insert({id, pylir::Syntax::Scope::NonLocal}).first;
                            if (iter->second == pylir::Syntax::Scope::Local)
                            {
                                iter->second = pylir::Syntax::Scope::Cell;
                            }
                        })
                    .template Case<pylir::Syntax::ClassDef*>(
                        [&id = id](pylir::Syntax::ClassDef* classDef) {
                            classDef->scope.identifiers.insert({id, pylir::Syntax::Scope::NonLocal});
                        });
            }
        }

        std::vector<const pylir::Syntax::Scope*> m_scopes;
        ScopeOwner m_parentDef;
        SemanticAnalysis* m_analysis;

    public:
        explicit NamespaceVisitor(SemanticAnalysis& analysis) : m_analysis(&analysis) {}

        using Visitor::visit;

        void visit(pylir::Syntax::ClassDef& classDef)
        {
            {
                pylir::ValueReset reset(m_parentDef);
                m_parentDef = &classDef;
                Visitor::visit(classDef);
            }
            finishNamespace(classDef.scope);
        }

        void visit(pylir::Syntax::FuncDef& funcDef)
        {
            {
                m_scopes.push_back(&funcDef.scope);
                auto exit = llvm::make_scope_exit([&] { m_scopes.pop_back(); });
                pylir::ValueReset reset(m_parentDef);
                m_parentDef = &funcDef;
                Visitor::visit(funcDef);
            }
            finishNamespace(funcDef.scope);
        }

        void visit(pylir::Syntax::Lambda& lambda)
        {
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
    llvm::TypeSwitch<ScopeOwner>(owner).Case<Syntax::Lambda*, Syntax::FuncDef*, Syntax::ClassDef*>(
        [&](auto* ptr) { visitor.visit(*ptr); });
}

void pylir::SemanticAnalysis::visit(Syntax::FuncDef& funcDef)
{
    for (auto& iter : funcDef.decorators)
    {
        visit(iter);
    }
    addToNamespace(funcDef.funcName);
    for (auto& iter : funcDef.parameterList)
    {
        visit(iter);
    }
    if (funcDef.maybeSuffix)
    {
        visit(*funcDef.maybeSuffix);
    }
    {
        ValueReset inLoopReset(m_inLoop);
        ValueReset inFuncReset(m_inFunc);
        ValueReset currentScopeOwnerReset(m_currentScopeOwner);
        m_inLoop = false;
        m_inFunc = true;
        m_currentScopeOwner = &funcDef;
        for (auto& iter : funcDef.parameterList)
        {
            addToNamespace(iter.name);
        }
        visit(*funcDef.suite);
    }
    if (!m_currentScopeOwner)
    {
        // this indicates that this funcdef is at global scope or only nested in classes. We now need to resolve any
        // nonlocals inside any nested funcDefs and figure out whether any unknowns are nonlocal or global

        for (auto& [iter, kind] : funcDef.scope.identifiers)
        {
            if (kind != Syntax::Scope::NonLocal)
            {
                continue;
            }
            createError(iter, Diag::COULD_NOT_FIND_VARIABLE_N_IN_OUTER_SCOPES, iter.getValue()).addHighlight(iter);
        }
        finishNamespace(&funcDef);
    }
}

void pylir::SemanticAnalysis::visit(Syntax::ClassDef& classDef)
{
    for (auto& iter : classDef.decorators)
    {
        visit(iter);
    }
    addToNamespace(classDef.className);
    if (classDef.inheritance)
    {
        for (auto& iter : classDef.inheritance->argumentList)
        {
            visit(iter);
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
    {
        finishNamespace(&classDef);
    }
}
