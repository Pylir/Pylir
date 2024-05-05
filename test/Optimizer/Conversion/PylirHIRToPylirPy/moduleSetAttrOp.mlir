// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK: #[[$MODULE:.*]] = #py.globalValue<module>
#module = #py.globalValue<module>

// CHECK-LABEL: py.func @test$impl(
// CHECK-SAME: %[[CLOSURE:[[:alnum:]]+]]
pyHIR.globalFunc @test(%closure) {
  // CHECK: %[[C:.*]] = constant(#[[$MODULE]])
  // CHECK: %[[INDEX:.*]] = arith.constant
  // CHECK: %[[DICT:.*]] = getSlot %[[C]][%[[INDEX]]]
  // CHECK: %[[KEY:.*]] = constant(#py.str<"foo">)
  // CHECK: %[[HASH:.*]] = str_hash %[[KEY]]
  // CHECK: dict_setItem %[[DICT]][%[[KEY]] hash(%[[HASH]])] to %[[CLOSURE]]
  module_setAttr #module["foo"] to %closure
  return %closure
}
