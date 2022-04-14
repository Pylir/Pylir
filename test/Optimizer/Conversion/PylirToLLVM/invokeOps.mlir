// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

func @foo() -> !py.dynamic {
    %0 = py.constant(#py.unbound)
    return %0 : !py.dynamic
}

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.BaseException = #py.type

func @invoke_test(%trueValue : !py.dynamic) -> !py.dynamic {
    %result = py.invoke @foo() : () -> !py.dynamic
        label ^success unwind ^failure

^success:
    return %trueValue : !py.dynamic

^failure(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: llvm.func @invoke_test
// CHECK-SAME: %[[TRUE_VALUE:[[:alnum:]]+]]
// CHECK-NEXT: %[[BASE_EXCEPTION:.*]] = llvm.mlir.addressof @builtins.BaseException
// CHECK-NEXT: %[[BIT_CAST:.*]] = llvm.bitcast %[[BASE_EXCEPTION]]
// CHECK-NEXT: %[[BIT_CAST:.*]] = llvm.bitcast %[[BASE_EXCEPTION]]
// CHECK-NEXT: llvm.invoke @foo() to ^[[HAPPY:.*]] unwind ^[[UNWIND:[[:alnum:]]+]]
// CHECK-NEXT: ^[[UNWIND]]:
// CHECK-NEXT: %[[LANDING_PAD:.*]] = llvm.landingpad
// CHECK-SAME: catch %[[BIT_CAST]]
// CHECK-NEXT: %[[EXCEPTION_HEADER_i8:.*]] = llvm.extractvalue %[[LANDING_PAD]][0 : i32]
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

