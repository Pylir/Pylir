#pragma once

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopeExit.h>

#include <pylir/Parser/Visitor.hpp>

namespace pylir
{
class GlobalDiscoverer : public Syntax::Visitor<GlobalDiscoverer>
{
    bool m_inGlobalScope = true;
    std::function<void(const IdentifierToken& global)> m_callback;

public:
    explicit GlobalDiscoverer(std::function<void(const IdentifierToken& global)> callback)
        : m_callback(std::move(callback))
    {
    }

    using Visitor::visit;

    void visit(const Syntax::FuncDef& funcDef);

    void visit(const Syntax::ClassDef& classDef);

    void visit(const Syntax::GlobalStmt& globalStmt);

    void visit(const Syntax::Target& target);

    void visit(const Syntax::AssignmentExpression& assignmentExpression);
};
} // namespace pylir
