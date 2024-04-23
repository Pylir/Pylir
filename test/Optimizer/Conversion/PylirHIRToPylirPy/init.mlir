// RUN: pylir-opt %s --convert-pylirHIR-to-pylirPy --split-input-file | FileCheck %s

// CHECK-LABEL: py.func @foo.__init__()
pyHIR.init "foo" {
  // CHECK-NEXT: return
  init_return
}

// CHECK-LABEL: py.func @__init__()
pyHIR.init "__main__" {
  // CHECK: call @foo.__init__() : () -> ()
  initModule @foo
  init_return
}
