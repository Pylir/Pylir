#include "GlobalDiscoverer.hpp"

#include <pylir/Support/ValueReset.hpp>

void pylir::GlobalDiscoverer::visit(const Syntax::FuncDef& funcDef)
{
    ValueReset reset(m_inGlobalScope, m_inGlobalScope);
    Visitor::visit(funcDef);
}

void pylir::GlobalDiscoverer::visit(const Syntax::ClassDef& classDef)
{
    ValueReset reset(m_inGlobalScope, m_inGlobalScope);
    Visitor::visit(classDef);
}

void pylir::GlobalDiscoverer::visit(const pylir::Syntax::GlobalStmt& globalStmt)
{
    m_callback(globalStmt.identifier);
    for (auto& [token, identifier] : globalStmt.rest)
    {
        (void)token;
        m_callback(identifier);
    }
    Visitor::visit(globalStmt);
}

void pylir::GlobalDiscoverer::visit(const pylir::Syntax::Target& target)
{
    if (auto* identifier = std::get_if<IdentifierToken>(&target.variant); identifier && m_inGlobalScope)
    {
        m_callback(*identifier);
    }
    Visitor::visit(target);
}

void pylir::GlobalDiscoverer::visit(const pylir::Syntax::AssignmentExpression& assignmentExpression)
{
    if (assignmentExpression.identifierAndWalrus)
    {
        m_callback(assignmentExpression.identifierAndWalrus->first);
    }
    Visitor::visit(assignmentExpression);
}
