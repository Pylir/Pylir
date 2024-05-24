// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK-LABEL: func @basic$impl(
// CHECK-SAME: %[[CLOSURE:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG3:[[:alnum:]]+]]
pyHIR.globalFunc @basic(%closure, %arg0, %arg1, %arg2, %arg3) {
  // CHECK: %[[NAME:.*]] = constant(#py.str<"Foo">)
  // CHECK: %[[BUILD_CLASS:.*]] = constant(#{{([[:alnum:]]|_)+}})
  // CHECK: %[[STR:.*]] = constant(#py.str<"test">)
  // CHECK: %[[HASH:.*]] = str_hash %[[STR]]
  // CHECK: %[[TUPLE:.*]] = makeTuple (%[[CLOSURE]], %[[NAME]], %[[ARG0]], *%[[ARG1]])
  // CHECK: %[[DICT:.*]] = makeDict (%[[STR]] hash(%[[HASH]]) : %[[ARG2]], **%[[ARG3]])
  // CHECK: call @pylir__call__(%[[BUILD_CLASS]], %[[TUPLE]], %[[DICT]])
  %0 = buildClass(%closure, "Foo", %arg0, *%arg1, "test"=%arg2, **%arg3)
  return %0
}
