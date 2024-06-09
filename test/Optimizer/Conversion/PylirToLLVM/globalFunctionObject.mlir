// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_object = #py.globalValue<builtins.object, initializer = #py.type>
py.external @builtins.object, #builtins_object
#builtins_function = #py.globalValue<builtins.function, initializer = #py.type>
py.external @builtins.function, #builtins_function
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_None = #py.globalValue<builtins.None, initializer = #py.type>
py.external @builtins.None, #builtins_None
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

#foo = #py.globalValue<foo, initializer = #py.function<@bar>>
py.external @foo, #foo

py.func @bar(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    return %arg0 : !py.dynamic
}

// CHECK-LABEL: @foo
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.function
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[TYPE]], %[[UNDEF]][0]
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.mlir.addressof @bar
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[ADDRESS]], %[[UNDEF1]][1]
// CHECK-NEXT: %[[CLOSURE_SIZE:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[UNDEF3:.*]] = llvm.insertvalue %[[CLOSURE_SIZE]], %[[UNDEF2]][2]
// CHECK-NEXT: llvm.return %[[UNDEF3]]
