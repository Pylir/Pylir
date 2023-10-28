; REQUIRES: x86-registered-target

; RUN: pylir --target=x86_64-pc-linux-gnu %s -S -emit-llvm -o - | FileCheck %s

target triple = "x86_64-w64-windows-gnu"

define i32 @square(i32) {
    %2 = mul nsw i32 %0, %0
    ret i32 %2
}

; CHECK-DAG: target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
; CHECK-DAG: target triple = "x86_64-w64-windows-gnu"
