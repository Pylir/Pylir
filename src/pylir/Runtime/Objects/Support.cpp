//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Support.hpp"

using namespace pylir::rt;

#include "Objects.hpp"

std::size_t PyObjectHasher::operator()(PyObject* object) const noexcept {
  return Builtins::Hash(*object).cast<PyInt>().to<std::size_t>();
}

bool PyObjectEqual::operator()(PyObject* lhs, PyObject* rhs) const noexcept {
  return *lhs == *rhs;
}

/// Read a uleb128 encoded value and advance pointer
/// See Variable Length Data Appendix C in:
/// @link http://dwarfstd.org/Dwarf4.pdf @unlink
/// @param data reference variable holding memory pointer to decode from
/// @returns decoded value
std::uintptr_t pylir::rt::readULEB128(const std::uint8_t** data) {
  std::uintptr_t result = 0;
  std::uintptr_t shift = 0;
  unsigned char byte;
  const uint8_t* p = *data;
  do {
    byte = *p++;
    result |= static_cast<std::uintptr_t>(byte & 0x7F) << shift;
    shift += 7;
  } while (byte & 0x80);
  *data = p;
  return result;
}

/// Read a sleb128 encoded value and advance pointer
/// See Variable Length Data Appendix C in:
/// @link http://dwarfstd.org/Dwarf4.pdf @unlink
/// @param data reference variable holding memory pointer to decode from
/// @returns decoded value
std::intptr_t pylir::rt::readSLEB128(const std::uint8_t** data) {
  std::uintptr_t result = 0;
  std::uintptr_t shift = 0;
  unsigned char byte;
  const uint8_t* p = *data;
  do {
    byte = *p++;
    result |= static_cast<std::uintptr_t>(byte & 0x7F) << shift;
    shift += 7;
  } while (byte & 0x80);
  *data = p;
  if ((byte & 0x40) && (shift < (sizeof(result) << 3)))
    result |= static_cast<std::uintptr_t>(~0) << shift;

  return static_cast<std::intptr_t>(result);
}
