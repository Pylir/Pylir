#pragma once

#include <fmt/format.h>

namespace pylir::Diag
{
template <class Formatting, class... Args>
void emitDiagnostics(const Formatting& formatting, Args&&... args)
{
}



} // namespace pylir::Diag
