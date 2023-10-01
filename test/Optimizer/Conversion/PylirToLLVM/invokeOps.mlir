// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @foo() -> !py.dynamic {
    %0 = constant(#py.unbound)
    return %0 : !py.dynamic
}

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @invoke_test(%trueValue : !py.dynamic) -> !py.dynamic {
    %result = invoke @foo() : () -> !py.dynamic
        label ^success unwind ^failure

^success:
    return %trueValue : !py.dynamic

^failure(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: llvm.func @invoke_test
// CHECK-SAME: %[[TRUE_VALUE:[[:alnum:]]+]]
// CHECK-NEXT: %[[BASE_EXCEPTION:.*]] = llvm.mlir.addressof @builtins.BaseException
// CHECK-NEXT: llvm.invoke @foo() to ^[[HAPPY:.*]] unwind ^[[UNWIND:[[:alnum:]]+]]
// CHECK-NEXT: ^[[UNWIND]]:
// CHECK-NEXT: %[[LANDING_PAD:.*]] = llvm.landingpad
// CHECK-SAME: catch %[[BASE_EXCEPTION]]
// CHECK-NEXT: %[[EXCEPTION_HEADER_i8:.*]] = llvm.extractvalue %[[LANDING_PAD]][0]
// CHECK-NEXT: %[[OFFSETOF:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[GEP:.*]] = llvm.sub %[[EXCEPTION_HEADER_i8]], %[[OFFSETOF]]
// CHECK-NEXT: %[[EXCEPTION_OBJECT:.*]] = llvm.inttoptr %[[GEP]]
// CHECK-NEXT: llvm.br ^[[DEST:[[:alnum:]]+]]
// CHECK-SAME: %[[EXCEPTION_OBJECT]]
// CHECK-NEXT: ^[[HAPPY]]:
// CHECK-NEXT: llvm.return %[[TRUE_VALUE]]
// CHECK-NEXT: ^[[DEST]]
// CHECK-SAME: %[[EXCEPTION_OBJECT:[[:alnum:]]+]]
// CHECK-NEXT: llvm.return %[[EXCEPTION_OBJECT]]

