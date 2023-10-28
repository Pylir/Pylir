; RUN: pylir %s -S -o - | FileCheck %s

; REQUIRES: x86-registered-target

; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

@g = constant i32 0, section "py_const"
@f = constant i32 0, section "py_const"
@e = constant i32 0, section "py_coll"
@d = constant i32 0, section "py_coll"
@c = constant i32 0, section "py_root"
@b = constant i32 0, section "py_root"
@a = constant i32 0, section "py_root"

; Required to get the GC plugin registered and emit the maps.
define void @dummy() gc "pylir-gc" {
    ret void
}

; CHECK: .internal pylir$roots
; CHECK-LABEL: pylir$roots:
; CHECK-NEXT: .quad a
; CHECK-NEXT: .quad b
; CHECK-NEXT: .quad c

; CHECK: .globl pylir_roots_start
; CHECK-LABEL: pylir_roots_start:
; CHECK-NEXT: .quad pylir$roots

; CHECK: .globl pylir_roots_end
; CHECK-LABEL: pylir_roots_end:
; CHECK-NEXT: .quad pylir$roots+24

; CHECK: .internal pylir$constants
; CHECK-LABEL: pylir$constants:
; CHECK-NEXT: .quad f
; CHECK-NEXT: .quad g

; CHECK: .globl pylir_constants_start
; CHECK-LABEL: pylir_constants_start:
; CHECK-NEXT: .quad pylir$constants

; CHECK: .globl pylir_constants_end
; CHECK-LABEL: pylir_constants_end:
; CHECK-NEXT: .quad pylir$constants+16

; CHECK: .internal pylir$collections
; CHECK-LABEL: pylir$collections:
; CHECK-NEXT: .quad d
; CHECK-NEXT: .quad e

; CHECK: .globl pylir_collections_start
; CHECK-LABEL: pylir_collections_start:
; CHECK-NEXT: .quad pylir$collections

; CHECK: .globl pylir_collections_end
; CHECK-LABEL: pylir_collections_end:
; CHECK-NEXT: .quad pylir$collections+16
