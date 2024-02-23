// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/Support/FileSystem.h>

#include <pylir/Diagnostics/DiagnosticsManager.hpp>

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace pylir {

struct CodeGenOptions {
  std::function<void(llvm::StringRef absoluteModule,
                     Diag::DiagnosticsDocManager* diagnostics,
                     std::pair<std::size_t, std::size_t> location)>
      moduleLoadCallback;
  std::string qualifier;
  bool implicitBuiltinsImport;
};
} // namespace pylir
