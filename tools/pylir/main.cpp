//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <llvm/Support/InitLLVM.h>

#include <pylir/Main/PylirMain.hpp>

int main(int argc, char** argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::setBugReportMsg(
      "PLEASE submit a bug report to https://github.com/zero9178/Pylir\n");
  return pylir::main(argc, argv);
}
