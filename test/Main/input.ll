; REQUIRES: x86-registered-target

; RUN: pylir %s -S -emit-llvm -o - | FileCheck %s

; ModuleID = 'LLVMDialectModule'

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

define i32 @square(i32) local_unnamed_addr #0 {
    %2 = mul nsw i32 %0, %0
    ret i32 %2
}

; CHECK-LABEL: define i32 @square
