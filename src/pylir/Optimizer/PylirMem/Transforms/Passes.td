// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIRMEM_TRANSFORMS_PASSES
#define PYLIRMEM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HeapToStackPass : Pass<"pylir-heap-to-stack"> {
  let dependentDialects = ["::pylir::Mem::PylirMemDialect"];

  let statistics = [
    Statistic<"m_heapAllocationsReplaced", "Heap allocations replaced",
      "Amount of heap allocations replaced with stack allocations">,
  ];

  let options = [
    Option<"m_maxObjectSize", "max-object-size", "std::size_t", "32",
        "Max amount of pointer sizes an object may have to still be allocated "
        "on the stack. Currently only counts trailing items.">,
  ];
}

#endif
