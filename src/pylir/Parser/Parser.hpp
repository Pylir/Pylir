
#pragma once

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Diagnostics/Document.hpp>
#include <pylir/Lexer/Lexer.hpp>

#include <tl/expected.hpp>

#include "Syntax.hpp"

namespace pylir
{
class Parser
{
    Lexer m_lexer;
    Lexer::iterator m_current;
    Diag::Document* m_document;

    tl::expected<Lexer::iterator, std::string> expect(TokenType tokenType);

public:
    explicit Parser(
        Diag::Document& document, int fileId = 0,
        std::function<void(Diag::DiagnosticsBuilder&& diagnosticsBuilder)> callBack = [](auto&&) {})
        : m_lexer(document, fileId, std::move(callBack)), m_current(m_lexer.begin()), m_document(&document)
    {
    }

    template <class T, class S, class... Args>
    [[nodiscard]] Diag::DiagnosticsBuilder createDiagnosticsBuilder(const T& location, const S& message, Args&&... args)
    {
        return Diag::DiagnosticsBuilder(*m_document, location, message, std::forward<Args>(args)...);
    }

    tl::expected<Syntax::Atom, std::string> parseAtom();

    tl::expected<Syntax::AttributeRef, std::string> parseAttributeRef();

    tl::expected<Syntax::Subscription, std::string> parseSubscription();

    tl::expected<Syntax::Slicing, std::string> parseSlicing();

    tl::expected<Syntax::Call, std::string> parseCall();

    tl::expected<Syntax::Primary, std::string> parsePrimary();

    tl::expected<Syntax::AwaitExpr, std::string> parseAwaitExpr();

    tl::expected<Syntax::Power, std::string> parsePower();

    tl::expected<Syntax::UExpr, std::string> parseUExpr();

    tl::expected<Syntax::MExpr, std::string> parseMExpr();

    tl::expected<Syntax::AExpr, std::string> parseAExpr();

    tl::expected<Syntax::ShiftExpr, std::string> parseShiftExpr();

    tl::expected<Syntax::AndExpr, std::string> parseAndExpr();

    tl::expected<Syntax::XorExpr, std::string> parseXorExpr();

    tl::expected<Syntax::OrExpr, std::string> parseOrExpr();

    tl::expected<Syntax::Comparison, std::string> parseComparison();

    tl::expected<Syntax::NotTest, std::string> parseNotTest();

    tl::expected<Syntax::AndTest, std::string> parseAndTest();

    tl::expected<Syntax::OrTest, std::string> parseOrTest();

    tl::expected<Syntax::AssignmentExpression, std::string> parseAssignmentExpression();

    tl::expected<Syntax::ConditionalExpression, std::string> parseConditionalExpression();

    tl::expected<Syntax::Expression, std::string> parseExpression();

    tl::expected<Syntax::LambdaExpression, std::string> parseLambdaExpression();

    tl::expected<Syntax::StarredExpression, std::string> parseStarredExpression();

    tl::expected<Syntax::CompFor, std::string> parseCompFor();

    tl::expected<Syntax::CompIf, std::string> parseCompIf();

    tl::expected<Syntax::Comprehension, std::string> parseComprehension();

    tl::expected<Syntax::Enclosure, std::string> parseEnclosure();
};
} // namespace pylir
