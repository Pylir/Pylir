// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/StringRef.h>

#include <optional>
#include <string>

namespace pylir
{
struct Version
{
    int majorVersion;
    int minorVersion;
    int patch;
    std::string suffix;
    std::string original;

    Version() : majorVersion(-1), minorVersion(-1), patch(-1) {}

    static std::optional<Version> parse(llvm::StringRef version);

    bool operator<(const Version& rhs) const
    {
        if (std::tie(majorVersion, minorVersion, patch) < std::tie(rhs.majorVersion, rhs.minorVersion, rhs.patch))
        {
            return true;
        }
        if (std::tie(majorVersion, minorVersion, patch) == std::tie(rhs.majorVersion, rhs.minorVersion, rhs.patch))
        {
            return suffix.empty() && !rhs.suffix.empty();
        }
        return false;
    }

    bool operator>(const Version& rhs) const
    {
        return rhs < *this;
    }

    explicit operator bool() const
    {
        return majorVersion != -1;
    }
};
} // namespace pylir
