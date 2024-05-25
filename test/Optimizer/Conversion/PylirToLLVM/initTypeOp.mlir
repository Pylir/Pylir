// RUN: pylir-opt %s -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s

// CHECK-LABEL: func @test
// CHECK-SAME: %[[MEMORY:[[:alnum:]]+]]
// CHECK-SAME: %[[NAME:[[:alnum:]]+]]
// CHECK-SAME: %[[MRO_MEMORY:[[:alnum:]]+]]
// CHECK-SAME: %[[MRO:[[:alnum:]]+]]
// CHECK-SAME: %[[SLOTS:[[:alnum:]]+]]
py.func @test(%memory: !pyMem.memory,
              %name : !py.dynamic,
              %mro_tuple_memory: !pyMem.memory,
              %mro : !py.dynamic,
              %slots : !py.dynamic) -> !py.dynamic {
  // CHECK: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.type
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
  // CHECK: llvm.store %[[TYPE]], %[[GEP]]

  // Code prepending the type to MRO is here.

  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 3]
  // CHECK: llvm.store %[[MRO_MEMORY]], %[[GEP]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 4]
  // CHECK: llvm.store %[[SLOTS]], %[[GEP]]

  // TODO: Layout and offset computation.

  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 5]
  // CHECK: %[[GEP2:.*]] = llvm.getelementptr %[[GEP]][0, 0]
  // CHECK: llvm.store %[[NAME]], %[[GEP2]]
  %0 = pyMem.initType %memory(name=%name, mro=%mro_tuple_memory to %mro, slots=%slots)
  return %0 : !py.dynamic
}
