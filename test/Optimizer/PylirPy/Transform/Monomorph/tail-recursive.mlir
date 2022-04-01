// RUN: pylir-opt %s --monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @foo(%arg0: !py.unknown) -> !py.unknown {
	%0 = py.constant(#py.int<1>) : !py.unknown
	%1 = py.int.cmp le %arg0, %0 : !py.unknown, !py.unknown
	py.cond_br %1, ^exit, ^recurse

^exit:
	py.return %0 : !py.unknown

^recurse:
	%2 = py.constant(#py.int<-1>) : !py.unknown
	%3 = py.int.add %2, %arg0 : !py.unknown, !py.unknown
	%4 = py.call @foo(%3) : (!py.class<@builtins.int>) -> !py.unknown
	py.return %4 : !py.unknown
}

func @__init__() {
	%0 = py.constant(#py.int<10>) : !py.unknown
	%1 = py.call @foo(%0) : (!py.unknown) -> !py.unknown
	test.use(%1) : !py.unknown
	py.return
}

// CHECK-LABEL: func @__init__()
// CHECK: py.call @[[FOO_CLONE:.*]](%{{.*}}) : (!py.class<@builtins.int>) -> !py.unknown

// This should be optimized better in the future, but for now lets just make sure it terminates

// CHECK: func private @[[FOO_CLONE]]
// CHECK-SAME: %{{.*}}: !py.class<@builtins.int>
// CHECK-SAME: -> !py.unknown
