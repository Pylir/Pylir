// RUN: pylir-opt %s --test-loop-info --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.int = #py.type

func @__init__() {
	%one = py.constant(#py.int<1>)
	%zero = py.constant(#py.int<0>)
	cf.br ^loop(%zero : !py.dynamic)

^loop(%iter : !py.dynamic):
	%0 = py.typeOf %iter
	%1 = py.type.mro %0
	%2, %found = py.mroLookup "__add__" in %1
	%3 = py.makeTuple (%iter, %one)
	%4 = py.constant(#py.dict<{}>)
	%5 = py.function.call %2(%2, %3, %4)
	%6 = test.random
	cf.cond_br %6, ^loop(%5 : !py.dynamic), ^exit

^exit:
	test.use(%5) : !py.dynamic
	py.unreachable
}

// CHECK: Loop at depth 0 containing: ^[[HEADER:.*]]<header>

// Header appears with a block argument later
// CHECK-LABEL: func @__init__()
// CHECK: ^[[HEADER]](%{{.*}}: !py.dynamic):

// -----

func @__init__() {
	cf.br ^headerOne

^headerOne:
	%0 = test.random
	cf.cond_br %0, ^headerTwo, ^exit

^headerTwo:
	%1 = test.random
	cf.cond_br %1, ^headerTwo, ^continue

^continue:
	cf.br ^headerOne

^exit:
	return
}

// CHECK: Loop at depth 0 containing: ^[[HEADER_ONE:.*]]<header>, ^[[HEADER_TWO:.*]], ^[[BB3:.*]]
// CHECK-NEXT: Loop at depth 1 containing: ^[[HEADER_TWO]]<header>

// Verify the blocks above match how they appear in the IR as well
// CHECK-LABEL: func @__init__()
// CHECK: ^[[HEADER_ONE]]:
// CHECK: ^[[HEADER_TWO]]:
// CHECK: ^[[BB3]]:

// -----

func @__init__() {
	cf.br ^headerOne

^headerOne:
	%0 = test.random
	cf.cond_br %0, ^headerTwo, ^exit

^headerTwo:
	%1 = test.random
	cf.cond_br %1, ^headerTwo, ^headerOne

^exit:
	return
}

// CHECK: Loop at depth 0 containing: ^[[HEADER_ONE:.*]]<header>, ^[[HEADER_TWO:.*]]
// CHECK-NEXT: Loop at depth 1 containing: ^[[HEADER_TWO]]<header>

// Verify the blocks above match how they appear in the IR as well
// CHECK-LABEL: func @__init__()
// CHECK: ^[[HEADER_ONE]]:
// CHECK: ^[[HEADER_TWO]]:
// CHECK: ^[[BB3]]:
