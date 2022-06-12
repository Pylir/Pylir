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

// -----

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.None = #py.type
py.globalValue const @builtins.str.__new__ = #py.type

func private @builtins.str.__new__$impl(!py.dynamic, !py.dynamic, !py.dynamic ,!py.dynamic) -> !py.dynamic

func private @builtins.print$impl(%arg0: !py.dynamic, %arg1: !py.dynamic, %arg2: !py.dynamic, %arg3: !py.dynamic) -> !py.dynamic {
  %0 = py.constant(#py.dict<{}>)
  %1 = py.constant(@builtins.str)
  %c0 = arith.constant 0 : index
  %2 = py.constant(#py.str<"">)
  %c1 = arith.constant 1 : index
  %3 = py.constant(#py.str<"\0A">)
  %4 = py.constant(#py.str<" ">)
  %5 = py.constant(@builtins.None)
  %6 = py.constant(@builtins.type)
  %7 = py.constant(@builtins.str.__new__)
  %8 = py.is %arg2, %5
  %9 = arith.select %8, %4, %arg2 : !py.dynamic
  %10 = py.is %arg3, %5
  %11 = arith.select %10, %3, %arg3 : !py.dynamic
  %12 = py.tuple.len %arg1
  %13 = arith.cmpi ult, %12, %c1 : index
  cf.cond_br %13, ^bb9(%2 : !py.dynamic), ^bb1
^bb1:  // pred: ^bb0
  %14 = py.tuple.getItem %arg1[%c0]
  %15 = py.is %1, %6
  cf.cond_br %15, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %16 = py.typeOf %14
  cf.br ^bb4(%16, %c1 : !py.dynamic, index)
^bb3:  // pred: ^bb1
  %17 = py.makeTuple (%14)
  %18 = py.call @builtins.str.__new__$impl(%7, %1, %17, %0) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  cf.br ^bb4(%18, %c1 : !py.dynamic, index)
^bb4(%19: !py.dynamic, %20: index):  // 3 preds: ^bb2, ^bb3, ^bb8
  %21 = arith.cmpi ult, %20, %12 : index
  cf.cond_br %21, ^bb5, ^bb9(%19 : !py.dynamic)
^bb5:  // pred: ^bb4
  %22 = py.tuple.getItem %arg1[%20]
  %23 = py.is %1, %6
  cf.cond_br %23, ^bb6, ^bb7
^bb6:  // pred: ^bb5
  %24 = py.typeOf %22
  cf.br ^bb8(%24 : !py.dynamic)
^bb7:  // pred: ^bb5
  %25 = py.makeTuple (%22)
  %26 = py.call @builtins.str.__new__$impl(%7, %1, %25, %0) : (!py.dynamic, !py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic
  cf.br ^bb8(%26 : !py.dynamic)
^bb8(%27: !py.dynamic):  // 2 preds: ^bb6, ^bb7
  %28 = py.str.concat %19, %9, %27
  %29 = arith.addi %20, %c1 : index
  cf.br ^bb4(%28, %29 : !py.dynamic, index)
^bb9(%30: !py.dynamic):  // 2 preds: ^bb0, ^bb4
  %31 = py.str.concat %30, %11
  py.intr.print %31
  return %5 : !py.dynamic
}

// CHECK: Loop at depth 0 containing: ^bb4<header>, ^bb5, ^bb7, ^bb6, ^bb8
