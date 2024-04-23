// RUN: pylir-opt %s --canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: globalFunc @test
pyHIR.globalFunc @test(%closure, %arg0) {
  // CHECK: %[[FALSE:.*]] = py.constant(#py.bool<False>)
  // CHECK: return %[[FALSE]]
  %0 = py.isUnboundValue %arg0
  %1 = py.bool_fromI1 %0
  return %1
}

// -----

// CHECK-LABEL: init "__main__"
pyHIR.init "__main__" {
  // CHECK: %[[FALSE:.*]] = py.constant(#py.bool<False>)

  // CHECK: func "test"
  %f = func "test"(%arg1) {
    // CHECK: return %[[FALSE]]
    %0 = py.isUnboundValue %arg1
    %1 = py.bool_fromI1 %0
    return %1
  }
  test.use(%f) : !py.dynamic
  init_return
}
