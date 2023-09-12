
//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "Toolchain.hpp"

namespace pylir {
class MSVCToolchain : public Toolchain {
protected:
  [[nodiscard]] bool defaultsToPIC() const override {
    return true;
  }

public:
  using Toolchain::Toolchain;

  [[nodiscard]] bool link(cli::CommandLine& commandLine,
                          llvm::StringRef objectFile) const override;
};
} // namespace pylir
