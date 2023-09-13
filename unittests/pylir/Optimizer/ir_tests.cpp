// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <mlir/IR/Block.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/MLIRContext.h>

#include <pylir/Optimizer/PylirPy/IR/PylirPyDialect.hpp>
#include <pylir/Optimizer/PylirPy/IR/PylirPyOps.hpp>

TEST_CASE("IR DictArgsIterator", "[IR]") {
  mlir::MLIRContext context;
  context.loadDialect<pylir::Py::PylirPyDialect>();
  auto loc = mlir::UnknownLoc::get(&context);
  mlir::OwningOpRef<pylir::Py::FuncOp> func =
      mlir::OpBuilder(&context).create<pylir::Py::FuncOp>(
          loc, "test",
          mlir::FunctionType::get(
              &context,
              std::vector<mlir::Type>(8, pylir::Py::DynamicType::get(&context)),
              {}));
  auto* block = func->addEntryBlock();

  auto builder = mlir::ImplicitLocOpBuilder::atBlockBegin(loc, block);
  auto vector = std::vector<pylir::Py::DictArg>{
      pylir::Py::DictEntry{block->getArgument(0), block->getArgument(1),
                           block->getArgument(2)},
      pylir::Py::MappingExpansion{block->getArgument(3)},
      pylir::Py::DictEntry{block->getArgument(4), block->getArgument(5),
                           block->getArgument(6)},
      pylir::Py::MappingExpansion{block->getArgument(7)},
  };
  auto op = builder.create<pylir::Py::MakeDictOp>(vector);

  auto range = op.getDictArgs();
  CHECK_THAT(std::vector(range.begin(), range.end()),
             Catch::Matchers::Equals(vector));
  auto reverse = llvm::reverse(range);
  std::reverse(vector.begin(), vector.end());
  CHECK_THAT(std::vector(reverse.begin(), reverse.end()),
             Catch::Matchers::Equals(vector));
}
