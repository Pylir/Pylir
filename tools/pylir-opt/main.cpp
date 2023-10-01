//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/ExternalModels/ExternalModels.hpp>
#include <pylir/Optimizer/Optimizer.hpp>
#include <pylir/Optimizer/PylirHIR/IR/PylirHIRDialect.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirMem/Transforms/Passes.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Passes.hpp>
#include <pylir/Optimizer/Transforms/Passes.hpp>

#include "Passes.hpp"
#include "TestDialect.hpp"

int main(int argc, char** argv) {
  mlir::registerTransformsPasses();
  mlir::registerReconcileUnrealizedCasts();
  mlir::registerArithToLLVMConversionPass();

  mlir::DialectRegistry registry;
  registry.insert<pylir::Mem::PylirMemDialect, pylir::Py::PylirPyDialect,
                  pylir::test::TestDialect, pylir::HIR::PylirHIRDialect,
                  mlir::DLTIDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                  mlir::func::FuncDialect>();

  pylir::registerExternalModels(registry);

  pylir::registerConversionPasses();
  pylir::registerTransformPasses();
  pylir::Py::registerTransformPasses();
  pylir::Mem::registerTransformsPasses();
  pylir::test::registerTestPasses();
  pylir::registerOptimizationPipelines();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "pylir-opt", registry));
}
