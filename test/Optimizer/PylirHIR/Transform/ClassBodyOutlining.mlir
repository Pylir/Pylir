// RUN: pylir-opt %s --pylir-class-body-outlining --split-input-file | FileCheck %s

// CHECK-LABEL: globalFunc @test(
// CHECK-SAME: %[[CLOSURE:[[:alnum:]]+]]
// CHECK-SAME: %[[KW:[[:alnum:]]+]]
pyHIR.globalFunc @test(%closure, %kw) {
  // CHECK: %[[FUNC:.*]] = func "__main__.Test"(%[[ARG0:.*]])
  // CHECK: py.dict_setItem %[[ARG0]][
  // CHECK: return
  // CHECK: %[[CLASS:.*]] = buildClass(%[[FUNC]], "__main__.Test", %[[CLOSURE]], "test"=%[[KW]])
  // CHECK: return %[[CLASS]]
  %0 = class "__main__.Test"(%closure, "test"=%kw) {
  ^bb0(%dict: !py.dynamic loc(unknown)):
    %0 = py.constant(#py.str<"__init__">)
    %1 = py.str_hash %0
    py.dict_setItem %dict[%0 hash(%1)] to %0
    class_return
  }
  return %0
}
