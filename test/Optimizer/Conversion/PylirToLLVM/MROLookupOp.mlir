// RUN: pylir-opt %s -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__slots__">)>}>
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.object = #py.type
py.globalValue @builtins.tuple = #py.type

func.func @test(%tuple : !py.dynamic) -> !py.dynamic {
    %0 = py.mroLookup "__call__" in %tuple
    return %0 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE]][0, 1]
// CHECK-NEXT: %[[LEN:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{.*}})
// CHECK-NEXT: llvm.br ^[[COND:.*]](%[[ZERO]] : i{{.*}})

// CHECK-NEXT: ^[[COND]](%[[ITER:.*]]: i{{.*}}):
// CHECK-NEXT: %[[LESS:.*]] = llvm.icmp "ult" %[[ITER]], %[[LEN]]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: llvm.cond_br %[[LESS]], ^[[BODY:.*]], ^[[EXIT:.*]](%[[NULL]] : {{.*}})
// CHECK-NEXT: ^[[BODY]]:
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE]][0, 2]
// CHECK-NEXT: %[[GEP_2:.*]] = llvm.getelementptr %[[GEP]][0, %[[ITER]]]
// CHECK-NEXT: %[[ELEMENT:.*]] = llvm.load %[[GEP_2]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ELEMENT]][0, 0]
// CHECK-NEXT: %[[TYPE:.*]] = llvm.load %[[GEP]]

// impl of getSlotOp follows...

// CHECK: %[[FOUND:.*]] = llvm.icmp "eq" %[[RES:.*]], %[[NULL]]
// CHECK-NEXT: llvm.cond_br %[[FOUND]], ^[[INCR:.*]], ^[[EXIT]](%[[RES]] : {{.*}})
// CHECK-NEXT: ^[[INCR]]:
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i{{.*}})
// CHECK-NEXT: %[[INCREMENTED:.*]] = llvm.add %[[ITER]], %[[ONE]]
// CHECK-NEXT: llvm.br ^[[COND]](%[[INCREMENTED]] : i{{.*}})
// CHECK-NEXT: ^[[EXIT]](%[[RES:.*]]: {{.*}}):
// CHECK-NEXT: llvm.return %[[RES]]
