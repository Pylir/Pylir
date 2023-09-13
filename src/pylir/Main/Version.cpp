//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Version.hpp"

#include <llvm/ADT/SmallVector.h>

std::optional<pylir::Version> pylir::Version::parse(llvm::StringRef version) {
  llvm::SmallVector<llvm::StringRef> splits;
  version.split(splits, '.');
  if (splits.empty())
    return {};

  Version result;
  result.original = version;
  if (splits[0].getAsInteger(10, result.majorVersion) ||
      result.majorVersion < 0)
    return {};

  if (splits.size() < 2)
    return result;

  if (splits.size() == 2) {
    if (auto end = splits[1].find_first_not_of("0123456789");
        end != llvm::StringRef::npos) {
      result.suffix = splits[1].substr(end);
      splits[1] = splits[1].substr(0, end);
    }
  }
  if (splits[1].getAsInteger(10, result.minorVersion) ||
      result.minorVersion < 0)
    return {};

  if (splits.size() < 3)
    return result;

  if (auto end = splits[2].find_first_not_of("0123456789");
      end != llvm::StringRef::npos) {
    result.suffix = splits[2].substr(end);
    splits[2] = splits[2].substr(0, end);
  }
  if (splits[2].getAsInteger(10, result.patch) || result.patch < 0)
    return {};

  return result;
}
