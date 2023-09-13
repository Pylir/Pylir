// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Path.h>
#include <llvm/TargetParser/Triple.h>

#include <string>
#include <vector>

namespace pylir {

enum class LinkerStyle {
  MSVC,  ///< Flags in the style of link.exe.
  ELF,   ///< Flags in the style of ld on ELF platforms.
  MinGW, ///< Flags in the style of ld on MinGW.
  Mac,   ///< Flags in the style of ld64.
  Wasm,  ///< Flags in the style of wasm-ld.
};

/// Abstraction for building the linker command for the toolchain.
/// It is essentially a list of strings that are passed to the linker. Its value
/// comes from the addition of 'add' methods for common options and concepts
/// used across all supported linkers, without having to strictly follow any
/// linkers concrete command line syntax. Follows the builder pattern for being
/// able to chain modification methods.
class LinkerInvocationBuilder {
  LinkerStyle m_linkerStyle;
  std::vector<std::string> m_args;

public:
  /// Constructs the builder using a particular linker style. This will be used
  /// for the final linker invocation as indicator which linker should be called
  /// and is also important for the higher level 'add' methods, to determine
  /// what command line syntax to use.
  explicit LinkerInvocationBuilder(LinkerStyle linkerStyle)
      : m_linkerStyle(linkerStyle) {}

  /// Returns the list of arguments added so far.
  [[nodiscard]] llvm::ArrayRef<std::string> getArgs() const {
    return m_args;
  }

  /// Returns the linker style this builder was initialized with.
  [[nodiscard]] LinkerStyle getLinkerStyle() const {
    return m_linkerStyle;
  }

  /// Adds a raw argument to the linker command if 'condition' is true.
  /// The argument is directly forwarded to the linker.
  LinkerInvocationBuilder& addArg(llvm::Twine argument, bool condition = true) {
    if (condition) {
      m_args.emplace_back(argument.str());
    }
    return *this;
  }

  /// Overload of the above to prevent accidentally using a string literal as
  /// condition.
  LinkerInvocationBuilder& addArg(llvm::Twine argument, const char*) = delete;

  /// Adds a list of raw arguments to the linker command that are directly
  /// forwarded to the linker.
  template <class... Args, std::enable_if_t<(sizeof...(Args) > 1)>* = nullptr>
  LinkerInvocationBuilder& addArgs(Args&&... args) {
    (m_args.emplace_back(args), ...);
    return *this;
  }

  /// Adds a list of raw arguments to the linker command that are directly
  /// forwarded to the linker.
  LinkerInvocationBuilder& addArgs(llvm::ArrayRef<std::string> arguments) {
    m_args.insert(m_args.end(), arguments.begin(), arguments.end());
    return *this;
  }

  /// Adds a list of raw arguments to the linker command that are directly
  /// forwarded to the linker.
  LinkerInvocationBuilder& addArgs(llvm::ArrayRef<const char*> arguments) {
    m_args.insert(m_args.end(), arguments.begin(), arguments.end());
    return *this;
  }

  /// Passes the given internal LLVM options to the linker.
  LinkerInvocationBuilder& addLLVMOptions(llvm::ArrayRef<std::string> options);

  /// Adds the emulation option to the linker, telling it about the target
  /// architecture. Only allowed for ELF and MinGW style!
  LinkerInvocationBuilder& addEmulation(const llvm::Triple& triple);

  /// Add a list of directories to the library search path of the linker.
  LinkerInvocationBuilder&
  addLibrarySearchDirs(llvm::ArrayRef<std::string> directories) {
    for (const auto& iter : directories)
      addLibrarySearchDir(iter);

    return *this;
  }

  /// Add a directory to the library search path of the linker.
  LinkerInvocationBuilder& addLibrarySearchDir(llvm::Twine directory);

  /// Add a directory, composed of 'args', interleaved with the native file
  /// separator, to the library search path of the linker.
  template <class First, class... Args,
            std::enable_if_t<(sizeof...(Args) > 0)>* = nullptr>
  LinkerInvocationBuilder& addLibrarySearchDir(First&& first, Args&&... args) {
    addLibrarySearchDir(
        (first + ... + (llvm::sys::path::get_separator() + args)));
    return *this;
  }

  /// Add the name of the output file to the linker command line.
  LinkerInvocationBuilder& addOutputFile(llvm::Twine outputFile);

  /// Add a library name that should be linked in to the command line.
  LinkerInvocationBuilder& addLibrary(llvm::Twine library);
};

} // namespace pylir
