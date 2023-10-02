// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__slots__">)>}>>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @test(%tuple : !py.dynamic) -> !py.dynamic {
    %c0 = arith.constant 0 : index
    %0 = mroLookup %c0 in %tuple
    return %0 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[TUPLE:[[:alnum:]]+]]
// CHECK-NEXT: %[[ZERO_I:.*]] = llvm.mlir.constant(0 : {{.*}}) : i{{[0-9]+}}
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE]][0, 1]
// CHECK-NEXT: %[[LEN:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{.*}})
// CHECK-NEXT: llvm.br ^[[COND:.*]](%[[ZERO]] : i{{.*}})

// CHECK-NEXT: ^[[COND]](%[[ITER:.*]]: i{{.*}}):
// CHECK-NEXT: %[[LESS:.*]] = llvm.icmp "ult" %[[ITER]], %[[LEN]]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.zero
// CHECK-NEXT: llvm.cond_br %[[LESS]], ^[[BODY:.*]], ^[[EXIT:.*]](%[[NULL]] : {{.*}})
// CHECK-NEXT: ^[[BODY]]:
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE]][0, 2]
// CHECK-NEXT: %[[GEP_2:.*]] = llvm.getelementptr %[[GEP]][0, %[[ITER]]]
// CHECK-NEXT: %[[ELEMENT:.*]] = llvm.load %[[GEP_2]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ELEMENT]][0, 0]
// CHECK-NEXT: %[[TYPE:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TYPE]][0, 1]
// CHECK-NEXT: %[[OFFSET:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[ADD:.*]] = llvm.add %[[OFFSET]], %[[ZERO_I]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ELEMENT]]
// CHECK-NEXT: %[[RES:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: %[[FOUND:.*]] = llvm.icmp "eq" %[[RES]], %[[NULL]]
// CHECK-NEXT: llvm.cond_br %[[FOUND]], ^[[INCR:.*]], ^[[EXIT]](%[[RES]] : {{.*}})
// CHECK-NEXT: ^[[INCR]]:
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i{{.*}})
// CHECK-NEXT: %[[INCREMENTED:.*]] = llvm.add %[[ITER]], %[[ONE]]
// CHECK-NEXT: llvm.br ^[[COND]](%[[INCREMENTED]] : i{{.*}})
// CHECK-NEXT: ^[[EXIT]](%[[RES:.*]]: {{.*}}):
// CHECK-NEXT: llvm.return %[[RES]]
