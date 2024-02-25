// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK-LABEL: py.func @foo.__init__() -> !py.dynamic
pyHIR.init "foo" {
  // CHECK-NEXT: %[[DICT:.*]] = makeDict
  // CHECK-NEXT: return %[[DICT]]
  %0 = py.makeDict ()
  init_return %0
}

// CHECK-LABEL: py.func @__init__() -> !py.dynamic
pyHIR.init "__main__" {
  %0 = py.makeDict ()
  // CHECK: call @foo.__init__()
  initModule @foo
  init_return %0
}
