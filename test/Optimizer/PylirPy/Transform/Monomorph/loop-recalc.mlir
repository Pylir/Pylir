// RUN: pylir-opt %s --pylir-monomorph --canonicalize --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.int = #py.type
py.globalValue const @builtins.tuple = #py.type

py.func @__init__() {
	%one = arith.constant 1 : index
	%zero = arith.constant 0 : index
	%random = constant(#py.int<69>)
	%tuple = makeTuple (%random)
	%len = py.tuple.len %tuple
	cf.br ^header(%zero : index)

^header(%iter : index):
	%0 = arith.cmpi ult, %iter, %len : index
	cf.cond_br %0, ^body, ^exit

^body:
    %1 = arith.addi %iter, %one : index
    cf.br ^header(%1 : index)

^exit:
	return
}

// CHECK-LABEL: func @__init__
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
// CHECK-NEXT: cf.br ^[[HEADER:.*]](%[[ZERO]] : index)
// CHECK-NEXT: ^[[HEADER]](%[[ITER:.*]]: index)
// CHECK-NEXT: %[[CMP:.*]] = arith.cmpi ult, %[[ITER]], %[[ONE]]
// CHECK-NEXT: cf.cond_br %[[CMP]], ^[[BODY:.*]], ^[[EXIT:.*]]
// CHECK-NEXT: ^[[BODY]]:
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[ITER]], %[[ONE]]
// CHECK-NEXT: cf.br ^[[HEADER]](%[[ADD]] : index)
// CHECK-NEXT: ^[[EXIT]]:
// CHECK-NEXT: return
