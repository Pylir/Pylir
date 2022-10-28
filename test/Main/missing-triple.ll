; RUN: pylir %s -S -emit-llvm -o - | FileCheck %s

define i32 @square(i32) {
    %2 = mul nsw i32 %0, %0
    ret i32 %2
}

; CHECK-DAG: target datalayout = "{{.*}}"
; CHECK-DAG: target triple = "{{.*}}"