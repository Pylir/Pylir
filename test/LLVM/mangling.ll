; REQUIRES: aarch64-registered-target

; RUN: pylir %s -S -o - | FileCheck %s

; Needed a target that has mangling. Mach-O targets always use `_` as prefix

target triple = "arm64-apple-darwin21.6.0"

@"builtins.next$handle" = private global ptr addrspace(1) null, section "__TEXT,py_root"

define void @__init__() gc "pylir-gc" {
  %statepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 2882400000, i32 0, ptr elementtype(void ()) @builtins.__init__, i32 0, i32 0, i32 0, i32 0) [ "deopt"() ]
  ret void
}

declare void @builtins.__init__()

declare token @llvm.experimental.gc.statepoint.p0(i64 immarg, i32 immarg, ptr, i32 immarg, i32 immarg, ...)

; CHECK: _pylir_stack_map
; CHECK: _pylir_roots_start
; CHECK: _pylir_roots_end
; CHECK: _pylir_constants_start
; CHECK: _pylir_constants_end
; CHECK: _pylir_collections_start
; CHECK: _pylir_collections_end
