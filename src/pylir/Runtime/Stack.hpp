#pragma once

#include <vector>

#include "Objects.hpp"

namespace pylir::rt
{
std::pair<std::uintptr_t, std::uintptr_t> collectStackRoots(std::vector<PyObject*>& results);
}
