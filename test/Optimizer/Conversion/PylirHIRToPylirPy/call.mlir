// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK-LABEL: func @basic$impl
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
pyHIR.globalFunc @basic(%arg0) {
  // CHECK: %[[K:.*]] = constant(#py.str<"k">)
  // CHECK: %[[HASH:.*]] = str_hash %[[K]]
  // CHECK: %[[TUPLE:.*]] = makeTuple (%[[ARG0]], *%[[ARG0]])
  // CHECK: %[[DICT:.*]] = makeDict (%[[K]] hash(%[[HASH]]) : %[[ARG0]], **%[[ARG0]])
  // CHECK: %[[RET:.*]] = call @pylir__call__(%[[ARG0]], %[[TUPLE]], %[[DICT]])
  %0 = call %arg0(%arg0, "k"=%arg0, *%arg0, **%arg0)
  // CHECK: return %[[RET]]
  return %0
}
