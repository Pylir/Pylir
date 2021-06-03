
#include "DiagnosticsBuilder.hpp"

#include <pylir/Support/Util.hpp>

#include <algorithm>
#include <numeric>
#include <set>
#include <unordered_map>

#include <fmt/color.h>
#include <fmt/format.h>

std::string pylir::Diag::DiagnosticsBuilder::printLine(std::size_t width, std::size_t lineNumber, Document& document,
                                                       std::vector<Label> labels)
{
    std::string result = fmt::format("{1: >{0}} | ", width, lineNumber);
    auto line = document.getLine(lineNumber);
    if (labels.empty())
    {
        result += Text::toUTF8String(line) + "\n";
        return result;
    }
    auto offset = line.data() - document.getText().data();
    {
        std::size_t lastEnd = 0;
        for (auto& iter : labels)
        {
            result += Text::toUTF8String(line.substr(lastEnd, iter.start - offset - lastEnd));
            fmt::text_style style;
            if (iter.optionalColour)
            {
                style |= fmt::fg(static_cast<fmt::color>(*iter.optionalColour));
            }
            if (iter.optionalEmphasis)
            {
                style |= static_cast<fmt::emphasis>(*iter.optionalEmphasis);
            }
            // If not pointing at newline
            if (iter.start - offset != line.size())
            {
                result += fmt::format(style, "{}",
                                      Text::toUTF8String(line.substr(iter.start - offset, iter.end - iter.start)));
            }
            lastEnd = iter.end - offset;
        }
        if (lastEnd <= line.size())
        {
            result += Text::toUTF8String(line.substr(lastEnd));
        }
    }
    result += '\n';

    result += fmt::format("{1: >{0}} | ", width, "");
    {
        std::size_t lastEnd = 0;
        std::u32string underlines;
        underlines.reserve(line.size());
        for (auto& iter : labels)
        {
            for (auto codepoint : line.substr(lastEnd, iter.start - offset - lastEnd))
            {
                if (Text::isWhitespace(codepoint))
                {
                    underlines += codepoint;
                    continue;
                }
                auto consoleWidth = Text::consoleWidth(codepoint);
                underlines.insert(underlines.end(), consoleWidth, U' ');
            }
            lastEnd = iter.end - offset;
            fmt::text_style style;
            if (iter.optionalColour)
            {
                style = fmt::fg(static_cast<fmt::color>(*iter.optionalColour));
            }
            if (iter.start - offset == line.size())
            {
                underlines += fmt::format(style, U"^");
                continue;
            }
            auto substr = line.substr(iter.start - offset, iter.end - iter.start);
            if (substr.size() == 1)
            {
                auto consoleWidth = Text::consoleWidth(substr.front());
                underlines += fmt::format(style, U"{0:^^{1}}", U"", consoleWidth);
                continue;
            }
            for (auto codepoint : substr)
            {
                auto consoleWidth = Text::consoleWidth(codepoint);
                underlines += fmt::format(style, U"{0:~^{1}}", U"", consoleWidth);
            }
        }
        result += Text::toUTF8String(underlines);
    }
    result += '\n';

    {
        result += fmt::format("{1: >{0}} | ", width, "");
        std::size_t lastEnd = 0;
        std::u32string underlines;
        underlines.reserve(line.size());
        for (auto& iter : labels)
        {
            fmt::text_style style;
            if (iter.optionalColour)
            {
                style = fmt::fg(static_cast<fmt::color>(*iter.optionalColour));
            }
            auto thisMid = (iter.end - iter.start) / 2 + iter.start - offset;
            for (auto codepoint : line.substr(lastEnd, thisMid - lastEnd))
            {
                if (Text::isWhitespace(codepoint))
                {
                    underlines += codepoint;
                    continue;
                }
                auto consoleWidth = Text::consoleWidth(codepoint);
                underlines.insert(underlines.end(), consoleWidth, U' ');
            }
            underlines += fmt::format(style, U"|");
            lastEnd = thisMid + 1;
        }
        result += Text::toUTF8String(underlines);
        result += '\n';
    }

    {
        labels.erase(std::remove_if(labels.begin(), labels.end(), [](const Label& label) { return !label.labelText; }),
                     labels.end());
        while (!labels.empty())
        {
            result += fmt::format("{1: >{0}} | ", width, "");
            std::size_t lastEnd = 0;
            std::u32string underlines;
            underlines.reserve(line.size());
            for (auto iter = labels.begin(); iter != labels.end();)
            {
                fmt::text_style style;
                if (iter->optionalColour)
                {
                    style = fmt::fg(static_cast<fmt::color>(*iter->optionalColour));
                }
                auto thisMid = (iter->end - iter->start) / 2 + iter->start - offset;
                for (auto codepoint : line.substr(lastEnd, thisMid - lastEnd))
                {
                    if (Text::isWhitespace(codepoint))
                    {
                        underlines += codepoint;
                        continue;
                    }
                    auto consoleWidth = Text::consoleWidth(codepoint);
                    underlines.insert(underlines.end(), consoleWidth, U' ');
                }
                auto text = Text::toUTF32String(*iter->labelText);
                std::size_t textWidth = std::accumulate(text.begin(), text.end(), (std::size_t)0,
                                                        [](std::size_t width, char32_t codepoint)
                                                        { return width + Text::consoleWidth(codepoint); });
                if (auto next = iter + 1; next != labels.end())
                {
                    std::size_t widthTillNext = 0;
                    auto nextMid = (next->end - next->start) / 2 + next->start - offset;
                    for (std::size_t i = thisMid; i < nextMid; i++)
                    {
                        widthTillNext += Text::consoleWidth(line[i]);
                    }
                    if (textWidth >= widthTillNext)
                    {
                        underlines += fmt::format(style, U"|");
                        underlines.insert(underlines.end(), widthTillNext - 1, U' ');
                        lastEnd = nextMid;
                        iter++;

                        continue;
                    }
                }
                underlines += fmt::format(style, U"{}", text);
                std::size_t widthTillEnd = 0;
                std::size_t i = thisMid;
                for (; widthTillEnd < textWidth && i < line.size(); i++)
                {
                    widthTillEnd += Text::consoleWidth(line[i]);
                }
                lastEnd = i;
                iter = labels.erase(iter);
            }
            result += Text::toUTF8String(underlines);
            result += '\n';
        }
    }

    return result;
}

std::string pylir::Diag::DiagnosticsBuilder::emitMessage(const Message& message, Severity severity) const
{
    auto& document = *message.document;
    auto [lineNumber, colNumber] = document.getLineCol(message.location);
    std::string_view severityStr;
    fmt::color colour;
    switch (severity)
    {
        case Warning:
            severityStr = "warning";
            colour = fmt::color::magenta;
            break;
        case Error:
            severityStr = "error";
            colour = fmt::color::red;
            break;
        case Note:
            severityStr = "note";
            colour = fmt::color::cyan;
            break;
        default: PYLIR_UNREACHABLE;
    }
    auto result = fmt::format(fmt::emphasis::bold, "{}:{}:{}: ", document.getFilename(), lineNumber, colNumber);
    result += fmt::format(fmt::emphasis::bold | fmt::fg(colour), "{}:", severityStr);
    result += fmt::format(fmt::emphasis::bold, " {}\n", message.message);

    struct LabelCompare
    {
        bool operator()(const Label& lhs, const Label& rhs) const noexcept
        {
            return lhs.start < rhs.start;
        }
    };

    std::unordered_map<std::size_t, std::set<Label, LabelCompare>> labeled;
    std::set<std::size_t> neededLines;
    neededLines.insert(lineNumber);
    for (auto& iter : message.labels)
    {
        auto end = document.getLineNumber(iter.end - 1);
        auto start = document.getLineNumber(iter.start);
        if (start == end)
        {
            labeled[start].insert(iter);
            neededLines.insert(start);
            continue;
        }

        auto midPos = (iter.end - iter.start) / 2 + iter.start;
        auto textLine = document.getLineNumber(midPos);
        {
            auto line = document.getLine(start);
            neededLines.insert(start);
            std::optional<std::string> labelText;
            if (textLine == start)
            {
                labelText = iter.labelText;
            }
            labeled[start].insert({iter.start,
                                   static_cast<std::size_t>(line.data() + line.size() - document.getText().data()),
                                   std::move(labelText), iter.optionalColour, iter.optionalEmphasis});
        }
        for (std::size_t i = start + 1; i < end; i++)
        {
            neededLines.insert(i);
            std::optional<std::string> labelText;
            if (textLine == i)
            {
                labelText = iter.labelText;
            }
            auto line = document.getLine(i);
            labeled[i].insert({static_cast<std::size_t>(line.data() - document.getText().data()),
                               static_cast<std::size_t>(line.data() + line.size() - document.getText().data()),
                               std::move(labelText), iter.optionalColour, iter.optionalEmphasis});
        }
        {
            auto line = document.getLine(end);
            neededLines.insert(end);
            std::optional<std::string> labelText;
            if (textLine == end)
            {
                labelText = iter.labelText;
            }
            labeled[end].insert({static_cast<std::size_t>(line.data() - document.getText().data()), iter.end,
                                 std::move(labelText), iter.optionalColour, iter.optionalEmphasis});
        }
    }
    auto largestLine = *std::prev(neededLines.end());
    auto width = pylir::roundUpTo(1 + (std::size_t)std::floor(log10f(largestLine)), 4);

    constexpr auto MARGIN = 1;
    for (std::size_t i = std::max<std::ptrdiff_t>(1, *neededLines.begin() - MARGIN); i < *neededLines.begin(); i++)
    {
        result += fmt::format("{1: >{0}} | {2}\n", width, i, Text::toUTF8String(document.getLine(i)));
    }
    for (std::size_t i : neededLines)
    {
        auto& set = labeled[i];
        result += printLine(width, i, document, {std::move_iterator(set.begin()), std::move_iterator(set.end())});
    }
    for (std::size_t i = largestLine + 1; i < largestLine + MARGIN + 1 && document.hasLine(i); i++)
    {
        result += fmt::format("{1: >{0}} | {2}\n", width, i, Text::toUTF8String(document.getLine(i)));
    }

    return result;
}
