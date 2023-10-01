// RUN: pylir-opt %s --test-loop-info --split-input-file | FileCheck %s

py.func @__init__() {
	%one = constant(#py.int<1>)
	%zero = constant(#py.int<0>)
	%c0 = arith.constant 0 : index
	cf.br ^loop(%zero : !py.dynamic)

^loop(%iter : !py.dynamic):
	%0 = typeOf %iter
	%1 = type_mro %0
	%2 = mroLookup %c0 in %1
	%3 = makeTuple (%iter, %one)
	%4 = constant(#py.dict<{}>)
	%5 = function_call %2(%2, %3, %4)
	%6 = test.random
	cf.cond_br %6, ^loop(%5 : !py.dynamic), ^exit

^exit:
	test.use(%5) : !py.dynamic
	py.unreachable
}

// CHECK: Loop at depth 0 containing: ^[[$HEADER:.*]]<header>

// Header appears with a block argument later
// CHECK-LABEL: func @__init__()
// CHECK: ^[[$HEADER]](%{{.*}}: !py.dynamic):

// -----

py.func @__init__() {
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

// CHECK: Loop at depth 0 containing: ^[[$HEADER_ONE:.*]]<header>, ^[[$HEADER_TWO:.*]], ^[[$BB3:.*]]
// CHECK-NEXT: Loop at depth 1 containing: ^[[$HEADER_TWO]]<header>

// Verify the blocks above match how they appear in the IR as well
// CHECK-LABEL: func @__init__()
// CHECK: ^[[$HEADER_ONE]]:
// CHECK: ^[[$HEADER_TWO]]:
// CHECK: ^[[$BB3]]:

// -----

py.func @__init__() {
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

// CHECK: Loop at depth 0 containing: ^[[$HEADER_ONE:.*]]<header>, ^[[$HEADER_TWO:.*]]
// CHECK-NEXT: Loop at depth 1 containing: ^[[$HEADER_TWO]]<header>

// Verify the blocks above match how they appear in the IR as well
// CHECK-LABEL: func @__init__()
// CHECK: ^[[$HEADER_ONE]]:
// CHECK: ^[[$HEADER_TWO]]:

