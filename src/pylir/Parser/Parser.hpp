
#pragma once

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>
#include <pylir/Diagnostics/Document.hpp>
#include <pylir/Lexer/Lexer.hpp>

#include <tcb/span.hpp>
#include <tl/expected.hpp>

#include "Syntax.hpp"

namespace pylir
{
class Parser
{
    Lexer m_lexer;
    Lexer::iterator m_current;
    Diag::Document* m_document;

    tl::expected<Token, std::string> expect(TokenType tokenType);

    bool lookaheadEquals(tcb::span<const TokenType> tokens);

    template <class ParseFunc, class CheckFunc>
    auto parseCommaList(
        ParseFunc parseFunc, CheckFunc checkFunc,
        std::optional<typename std::invoke_result_t<ParseFunc>::value_type>&& optionalFirst = std::nullopt)
        -> tl::expected<Syntax::CommaList<typename std::invoke_result_t<ParseFunc>::value_type>, std::string>
    {
        using T = typename std::invoke_result_t<ParseFunc>::value_type;
        if (!optionalFirst)
        {
            auto first = parseFunc();
            if (!first)
            {
                return tl::unexpected{std::move(first).error()};
            }
            optionalFirst = std::move(*first);
        }
        std::vector<std::pair<Token, std::unique_ptr<T>>> rest;
        std::optional<Token> last;
        while (m_current != m_lexer.end() && m_current->getTokenType() == TokenType::Comma)
        {
            auto comma = m_current++;
            if (!checkFunc(m_current->getTokenType()))
            {
                last = *comma;
                break;
            }
            auto other = parseFunc();
            if (!other)
            {
                return tl::unexpected{std::move(other).error()};
            }
            rest.emplace_back(*comma, std::make_unique<T>(*std::move(other)));
        }
        return Syntax::CommaList<T>{std::make_unique<T>(std::move(*optionalFirst)), std::move(rest), std::move(last)};
    }

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

    tl::expected<Syntax::AttributeRef, std::string> parseAttributeRef(std::unique_ptr<Syntax::Primary>&& primary);

    tl::expected<Syntax::Slicing, std::string> parseSlicing(std::unique_ptr<Syntax::Primary>&& primary);

    tl::expected<Syntax::Call, std::string> parseCall(std::unique_ptr<Syntax::Primary>&& primary);

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

    tl::expected<Syntax::CommaList<Syntax::Expression>, std::string> parseExpressionList();

    tl::expected<Syntax::LambdaExpression, std::string> parseLambdaExpression();

    tl::expected<Syntax::StarredExpression, std::string> parseStarredExpression();

    tl::expected<Syntax::CompFor, std::string> parseCompFor();

    tl::expected<Syntax::CompIf, std::string> parseCompIf();

    tl::expected<Syntax::Comprehension, std::string>
        parseComprehension(Syntax::AssignmentExpression&& assignmentExpression);

    tl::expected<Syntax::Enclosure, std::string> parseEnclosure();
};
} // namespace pylir
