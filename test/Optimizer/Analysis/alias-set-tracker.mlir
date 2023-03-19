// RUN: pylir-opt %s --test-alias-set-tracker --split-input-file | FileCheck %s

py.func @basic(%x : !py.dynamic, %y : !py.dynamic, %i : index) {
	%l1 = makeList ()
	%l2 = makeList ()
	%b = test.random
	cf.cond_br %b, ^succ(%l1 : !py.dynamic), ^succ(%l2 : !py.dynamic)

^succ(%l: !py.dynamic):
	return
}

// CHECK-LABEL: Alias sets for basic
// CHECK-NEXT: {
// CHECK-SAME: %arg0 %arg1
// CHECK-SAME: }
// CHECK-NEXT: {
// CHECK-SAME: %0
// CHECK-SAME: %3
// CHECK-SAME: %1
// CHECK-SAME: }
