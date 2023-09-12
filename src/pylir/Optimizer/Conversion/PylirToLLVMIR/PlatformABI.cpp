//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PlatformABI.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <pylir/Support/Macros.hpp>

std::size_t pylir::PlatformABI::getAlignOf(mlir::Type type) const {
  return m_dataLayout.getTypeABIAlignment(type);
}

std::size_t pylir::PlatformABI::getSizeOf(mlir::Type type) const {
  return m_dataLayout.getTypeSize(type);
}
