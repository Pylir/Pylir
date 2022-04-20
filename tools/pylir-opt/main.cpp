// Copyright 2022 Markus Böck
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <pylir/Optimizer/Conversion/Passes.hpp>
#include <pylir/Optimizer/PylirMem/IR/PylirMemDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/Transforms/Passes.hpp>
#include <pylir/Optimizer/Transforms/Passes.hpp>

#include "TestDialect.hpp"

std::unique_ptr<mlir::Pass> createTestMemorySSA();

std::unique_ptr<mlir::Pass> createTestInlinerInterface();

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

int main(int argc, char** argv)
{
    mlir::registerAllPasses();

    mlir::DialectRegistry registry;
    registry.insert<pylir::Mem::PylirMemDialect, pylir::Py::PylirPyDialect, pylir::test::TestDialect>();
    mlir::registerAllDialects(registry);

    pylir::registerConversionPasses();
    pylir::registerTransformPasses();
    pylir::Py::registerTransformPasses();
    ::registerTestPasses();

    return mlir::failed(mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
