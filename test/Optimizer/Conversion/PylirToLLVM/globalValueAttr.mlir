// RUN: pylir-opt %s -convert-pylir-to-llvm | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>

// CHECK-LABEL: func @foo()
py.func @foo() -> !py.dynamic {
    // CHECK: llvm.mlir.addressof @[[$SYMBOL:.*]] : !llvm.ptr
    %0 = py.constant(#builtins_int)
    return %0 : !py.dynamic
}

// CHECK: llvm.mlir.global internal @[[$SYMBOL]]()
