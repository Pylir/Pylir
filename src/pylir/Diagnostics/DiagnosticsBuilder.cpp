
#include "DiagnosticsBuilder.hpp"

#include <pylir/Support/Util.hpp>

#include <algorithm>
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
    result += Text::toUTF8String(line.substr(0, labels.front().start - offset));
    for (auto& iter : labels)
    {
        fmt::text_style style;
        if (iter.colour)
        {
            style |= fmt::fg(static_cast<fmt::color>(*iter.colour));
        }
        if (iter.emphasis)
        {
            style |= static_cast<fmt::emphasis>(*iter.emphasis);
        }
        // If not pointing at newline
        if (iter.start - offset != line.size())
        {
            result +=
                fmt::format(style, "{}", Text::toUTF8String(line.substr(iter.start - offset, iter.end - iter.start)));
        }
    }
    result += '\n';

    result += fmt::format("{1: >{0}} | ", width, "");

    for (auto& iter : labels)
    {
    }

    result += '\n';
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
    }
    auto result = fmt::format(fmt::emphasis::bold, "{}:{}:{} ", document.getFilename(), lineNumber, colNumber);
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

        auto textLine = (end - start) / 2 + start;
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
                                   std::move(labelText), iter.colour, iter.emphasis});
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
                               std::move(labelText), iter.colour, iter.emphasis});
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
                                 std::move(labelText), iter.colour, iter.emphasis});
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

    return result;
}
