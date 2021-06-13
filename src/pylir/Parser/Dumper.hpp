#pragma once

#include <string>

#include "Syntax.hpp"

namespace pylir
{
class Dumper
{
    std::size_t m_level = 0;

    std::string addMiddleChild(std::string&& middleChildDump);

    std::string addLastChild(std::string&& lastChildDump);

public:
    std::string dump(const Syntax::Atom& atom);

    std::string dump(const Syntax::Enclosure& enclosure);

    std::string dump(const Syntax::Primary& primary);

    std::string dump(const Syntax::AttributeRef& attribute);

    std::string dump(const Syntax::Subscription& subscription);

    std::string dump(const Syntax::Slicing& slicing);

    std::string dump(const Syntax::Comprehension& comprehension);

    std::string dump(const Syntax::AssignmentExpression& assignmentExpression);

    std::string dump(const Syntax::Call& call);

    std::string dump(const Syntax::AwaitExpr& awaitExpr);

    std::string dump(const Syntax::UExpr& uExpr);

    std::string dump(const Syntax::Power& power);

    std::string dump(const Syntax::MExpr& mExpr);

    std::string dump(const Syntax::AExpr& aExpr);

    std::string dump(const Syntax::ShiftExpr& shiftExpr);

    std::string dump(const Syntax::AndExpr& andExpr);

    std::string dump(const Syntax::XorExpr& xorExpr);

    std::string dump(const Syntax::OrExpr& orExpr);

    std::string dump(const Syntax::Comparison& comparison);

    std::string dump(const Syntax::NotTest& notTest);

    std::string dump(const Syntax::AndTest& andTest);

    std::string dump(const Syntax::OrTest& orTest);

    std::string dump(const Syntax::ConditionalExpression& conditionalExpression);

    std::string dump(const Syntax::LambdaExpression& lambdaExpression);

    std::string dump(const Syntax::Expression& expression);

    std::string dump(const Syntax::StarredItem& starredItem);

    std::string dump(const Syntax::StarredExpression& starredExpression);

    std::string dump(const Syntax::CompIf& compIf);

    std::string dump(const Syntax::CompFor& compFor);
};

} // namespace pylir
