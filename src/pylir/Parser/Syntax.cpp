#include "Syntax.hpp"

#include <pylir/Diagnostics/DiagnosticsBuilder.hpp>

std::pair<std::size_t, std::size_t>
    pylir::Diag::LocationProvider<pylir::Syntax::Enclosure, void>::getRange(const Syntax::Enclosure& value) noexcept
{
    return pylir::match(
        value.variant,
        [](const Syntax::Enclosure::ParenthForm& parenthForm) -> std::pair<std::size_t, std::size_t> {
            return {LocationProvider<BaseToken>::getRange(parenthForm.openParenth).first,
                    LocationProvider<BaseToken>::getRange(parenthForm.closeParenth).second};
        },
        [](const Syntax::Enclosure::GeneratorExpression& parenthForm) -> std::pair<std::size_t, std::size_t> {
            return {LocationProvider<BaseToken>::getRange(parenthForm.openParenth).first,
                    LocationProvider<BaseToken>::getRange(parenthForm.closeParenth).second};
        },
        [](const Syntax::Enclosure::YieldAtom& parenthForm) -> std::pair<std::size_t, std::size_t> {
            return {LocationProvider<BaseToken>::getRange(parenthForm.openParenth).first,
                    LocationProvider<BaseToken>::getRange(parenthForm.closeParenth).second};
        },
        [](const Syntax::Enclosure::ListDisplay& parenthForm) -> std::pair<std::size_t, std::size_t> {
            return {LocationProvider<BaseToken>::getRange(parenthForm.openSquare).first,
                    LocationProvider<BaseToken>::getRange(parenthForm.closeSquare).second};
        },
        [](const Syntax::Enclosure::SetDisplay& parenthForm) -> std::pair<std::size_t, std::size_t> {
            return {LocationProvider<BaseToken>::getRange(parenthForm.openBrace).first,
                    LocationProvider<BaseToken>::getRange(parenthForm.closeBrace).second};
        },
        [](const Syntax::Enclosure::DictDisplay& parenthForm) -> std::pair<std::size_t, std::size_t> {
            return {LocationProvider<BaseToken>::getRange(parenthForm.openBrace).first,
                    LocationProvider<BaseToken>::getRange(parenthForm.closeBrace).second};
        });
}
