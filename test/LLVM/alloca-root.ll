; RUN: pylir %s -emit-llvm -S -o - | FileCheck %s

declare void @builtins.__init__()

declare token @llvm.experimental.gc.statepoint.p0(i64 immarg, i32 immarg, ptr, i32 immarg, i32 immarg, ...)

declare void @escape(ptr addrspace(1)) "gc-leaf-function"

define void @__init__() gc "pylir-gc" {
    %a = alloca { ptr addrspace(1), [0 x ptr addrspace(1)] }, addrspace(1)
    call void @builtins.__init__()
    call void @escape(ptr addrspace(1) %a)
    ret void
}

; CHECK-LABEL: define void @__init__()
; CHECK-NEXT: %[[ALLOCA:.*]] = alloca
; CHECK-NEXT: %{{.*}} = call token ({{.*}}) @llvm.experimental.gc.statepoint{{.*}}({{.*}}@builtins.__init__{{.*}}) [ "deopt"(ptr addrspace(1) %[[ALLOCA]]) ]
