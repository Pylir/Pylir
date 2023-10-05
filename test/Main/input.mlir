// RUN: pylir %s -o - -S -emit-llvm | FileCheck %s
// RUN: pylir-opt %s -o %t.mlirbc -emit-bytecode
// RUN: pylir %t.mlirbc -o - -S -emit-llvm | FileCheck %s

py.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = typeOf %arg0
    %1 = typeOf %0
    %c0 = arith.constant 0 : index
    %2 = getSlot %0[%c0]
    return %2 : !py.dynamic
}

// CHECK-LABEL: define {{.*}} ptr addrspace({{[0-9]+}}) @foo(ptr addrspace({{[0-9]+}}) %{{.*}})
