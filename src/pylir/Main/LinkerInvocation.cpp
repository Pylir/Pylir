// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LinkerInvocation.hpp"

#include <llvm/Support/FileSystem.h>

#include <pylir/Support/Macros.hpp>

pylir::LinkerInvocationBuilder&
pylir::LinkerInvocationBuilder::addEmulation(const llvm::Triple& triple) {
  PYLIR_ASSERT(m_linkerStyle == LinkerStyle::ELF ||
               m_linkerStyle == LinkerStyle::MinGW);
  m_args.emplace_back("-m");
  if (m_linkerStyle == LinkerStyle::MinGW) {
    switch (triple.getArch()) {
    case llvm::Triple::x86: m_args.emplace_back("i386pe"); break;
    case llvm::Triple::x86_64: m_args.emplace_back("i386pep"); break;
    case llvm::Triple::aarch64: m_args.emplace_back("arm64pe"); break;
    default: PYLIR_UNREACHABLE;
    }
  } else {
    switch (triple.getArch()) {
    case llvm::Triple::x86_64: m_args.emplace_back("elf_x86_64"); break;
    default: PYLIR_UNREACHABLE;
    }
  }
  return *this;
}

pylir::LinkerInvocationBuilder& pylir::LinkerInvocationBuilder::addLLVMOptions(
    llvm::ArrayRef<std::string> options) {
  switch (m_linkerStyle) {
  case LinkerStyle::ELF:
    for (const auto& iter : options)
      m_args.push_back("--mllvm=" + iter);

    break;
  case LinkerStyle::Mac:
    for (const auto& iter : options) {
      m_args.emplace_back("-mllvm");
      m_args.push_back(iter);
    }
    break;
  case LinkerStyle::MinGW:
    for (const auto& iter : options)
      m_args.push_back("--Xlink=/mllvm:" + iter);

    break;

  case LinkerStyle::MSVC:
    for (const auto& iter : options)
      m_args.push_back("/mllvm:" + iter);

    break;
  default: PYLIR_UNREACHABLE;
  }

  return *this;
}

pylir::LinkerInvocationBuilder&
pylir::LinkerInvocationBuilder::addLibrarySearchDir(llvm::Twine directory) {
  switch (m_linkerStyle) {
  case LinkerStyle::MSVC:
    m_args.push_back(("-libpath:" + directory).str());
    break;
  case LinkerStyle::Mac:
  case LinkerStyle::MinGW:
  case LinkerStyle::ELF: m_args.push_back(("-L" + directory).str()); break;
  default: PYLIR_UNREACHABLE;
  }
  return *this;
}

pylir::LinkerInvocationBuilder&
pylir::LinkerInvocationBuilder::addOutputFile(llvm::Twine outputFile) {
  switch (m_linkerStyle) {
  case LinkerStyle::MSVC:
    m_args.emplace_back(("-out:" + outputFile).str());
    break;
  case LinkerStyle::Mac:
  case LinkerStyle::MinGW:
  case LinkerStyle::ELF:
    m_args.emplace_back("-o");
    m_args.emplace_back(outputFile.str());
    break;
  default: PYLIR_UNREACHABLE;
  }
  return *this;
}

pylir::LinkerInvocationBuilder&
pylir::LinkerInvocationBuilder::addLibrary(llvm::Twine library) {
  switch (m_linkerStyle) {
  case LinkerStyle::MSVC: {
    auto str = library.str();
    if (!llvm::StringRef(str).ends_with(".lib"))
      str += ".lib";

    m_args.push_back(std::move(str));
    break;
  }
  case LinkerStyle::Mac:
  case LinkerStyle::MinGW:
  case LinkerStyle::ELF: m_args.emplace_back(("-l" + library).str()); break;
  default: PYLIR_UNREACHABLE;
  }
  return *this;
}
