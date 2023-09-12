//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-reduce/MlirReduceMain.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Passes.hpp>
#include <pylir/Optimizer/Transforms/Passes.hpp>

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  pylir::registerConversionPasses();
  pylir::registerTransformPasses();
  pylir::Py::registerTransformPasses();

  mlir::DialectRegistry registry;
  registry.insert<pylir::Mem::PylirMemDialect, pylir::Py::PylirPyDialect>();
  mlir::registerAllDialects(registry);
  mlir::MLIRContext context(registry);

  return mlir::failed(mlir::mlirReduceMain(argc, argv, context));
}
