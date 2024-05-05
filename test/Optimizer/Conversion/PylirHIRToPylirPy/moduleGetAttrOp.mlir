// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK: #[[$MODULE:.*]] = #py.globalValue<module>
#module = #py.globalValue<module>

// CHECK-LABEL: py.func @test$impl(
pyHIR.globalFunc @test(%closure) {
  // CHECK: %[[C:.*]] = constant(#[[$MODULE]])
  // CHECK: %[[INDEX:.*]] = arith.constant
  // CHECK: %[[DICT:.*]] = getSlot %[[C]][%[[INDEX]]]
  // CHECK: %[[KEY:.*]] = constant(#py.str<"foo">)
  // CHECK: %[[HASH:.*]] = str_hash %[[KEY]]
  // CHECK: %[[RET:.*]] = dict_tryGetItem %[[DICT]][%[[KEY]] hash(%[[HASH]])]
  %0 = module_getAttr #module["foo"]
  // CHECK: return %[[RET]]
  return %0
}
