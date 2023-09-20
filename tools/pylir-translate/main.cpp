//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/InitAllTranslations.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>

int main(int argc, char** argv) {
  mlir::registerFromLLVMIRTranslation();
  mlir::registerToLLVMIRTranslation();

  return failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
