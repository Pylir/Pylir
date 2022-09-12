// RUN: pylir-opt %s -pass-pipeline='pylir-fixpoint{optimization-pipeline=test-add-change max-iteration-count=10}' | FileCheck %s

// CHECK-COUNT-10: test.change
