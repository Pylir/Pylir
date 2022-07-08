// RUN: pylir-opt %s --convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.object = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @one = #py.type

func.func @linear_search(%tuple : !py.dynamic) -> i1 {
    %0 = py.constant(@one)
    %1 = py.tuple.contains %0 in %tuple
    return %1 : i1
}

// CHECK-LABEL: @linear_search
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[ADDR:.*]] = llvm.mlir.addressof @one
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[ZERO]], 1]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: llvm.br ^[[CONDITION:[[:alnum:]]+]]
// CHECK-SAME: %[[ZERO]]
// CHECK-NEXT: ^[[CONDITION]]
// CHECK-SAME: %[[ITER:[[:alnum:]]+]]
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "ne" %[[ITER]], %[[SIZE]]
// CHECK-NEXT: llvm.cond_br %[[CMP]], ^[[BODY:.*]], ^[[EXIT:.*]](%[[CMP]] : i1)
// CHECK-NEXT: ^[[BODY]]:
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[ZERO]], 2]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP]][%[[ZERO]], %[[ITER]]]
// CHECK-NEXT: %[[LOAD:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: %[[CMP:.*]] = llvm.icmp "eq" %[[LOAD]], %[[ADDR]]
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : index)
// CHECK-NEXT: %[[INCREMENTED:.*]] = llvm.add %[[ITER]], %[[ONE]]
// CHECK-NEXT: %[[TRUE:.*]] = llvm.mlir.constant(true)
// CHECK-NEXT: llvm.cond_br %[[CMP]], ^[[EXIT]](%[[TRUE]] : i1), ^[[CONDITION]](%[[INCREMENTED]] : i{{.*}})
// CHECK-NEXT: ^[[EXIT]]
// CHECK-SAME: %[[RESULT:[[:alnum:]]+]]
// CHECK-NEXT: llvm.return %[[RESULT]]
