; REQUIRES: aarch64-registered-target

; RUN: pylir %s -S -o - | FileCheck %s

target triple = "arm64-apple-darwin21.6.0"

@"builtins.next$handle" = private global ptr addrspace(1) null

define void @foo(ptr addrspace(1) %0) gc "pylir-gc" {
  %statepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 2882400000, i32 0, ptr elementtype(void ()) @builtins.__init__, i32 0, i32 0, i32 0, i32 0) [ "deopt"(ptr addrspace(1) %0) ]
  ret void
}

; CHECK-LABEL: _foo
; CHECK: {{^}}L[[$LABEL:.*]]:{{[[:space:]]}}

declare void @builtins.__init__()

declare token @llvm.experimental.gc.statepoint.p0(i64 immarg, i32 immarg, ptr, i32 immarg, i32 immarg, ...)

; CHECK-LABEL: pylir_stack_map:
; Magic PYLR
; CHECK-NEXT: .long 1348029522
; CHECK: .p2align 3
; CHECK-NEXT: .quad _foo+(L[[$LABEL]]-_foo)
