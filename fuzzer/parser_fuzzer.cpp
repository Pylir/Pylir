//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pylir/Parser/Parser.hpp>

#include <cstdint>
#include <cstring>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                      std::size_t size) {
  std::string input(size, '\0');
  std::memcpy(input.data(), data, size);

  pylir::Diag::DiagnosticsManager manager;
  pylir::Diag::Document document(input);
  auto docManager = manager.createSubDiagnosticManager(document);
  pylir::Parser parser(docManager);

  (void)parser.parseFileInput();

  return 0;
}
