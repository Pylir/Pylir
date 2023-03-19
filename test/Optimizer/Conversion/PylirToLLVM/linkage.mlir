// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @__init__() {
    return
}

// CHECK: llvm.func @__init__

py.func private @impl() {
    return
}

// CHECK: llvm.func internal @impl

