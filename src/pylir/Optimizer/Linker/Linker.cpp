//  Licensed under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Linker.hpp"

#include <pylir/Support/Macros.hpp>

mlir::OwningOpRef<mlir::ModuleOp> pylir::linkModules(
    llvm::MutableArrayRef<mlir::OwningOpRef<mlir::ModuleOp>> modules) {
  PYLIR_ASSERT(!modules.empty());
  auto first = std::move(modules.front());
  auto rest = modules.drop_front();

  mlir::SymbolTable table(*first);
  for (auto& iter : rest) {
    auto& operations = first->getBody()->getOperations();
    auto& back = operations.back();
    operations.splice(operations.end(), iter->getBody()->getOperations());
    for (auto& newOps : llvm::make_early_inc_range(llvm::make_range(
             std::next(back.getIterator()), operations.end()))) {
      auto symbol = mlir::dyn_cast<mlir::SymbolOpInterface>(newOps);
      if (!symbol)
        continue;

      auto existing = table.lookup<mlir::SymbolOpInterface>(symbol.getName());
      if (!existing) {
        table.insert(symbol);
        continue;
      }
      // If both are declarations, prefer to keep the existing one.
      if (symbol.isDeclaration()) {
        symbol->erase();
        continue;
      }
      PYLIR_ASSERT(existing.isDeclaration());
      table.erase(existing);
      table.insert(symbol);
    }
  }
  return first;
}
