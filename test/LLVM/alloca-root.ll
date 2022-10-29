; RUN: pylir %s -emit-llvm -S -o - | FileCheck %s

declare void @builtins.__init__()

declare void @llvm.lifetime.start.p1(i64, ptr addrspace(1) nocapture)

declare void @escape(ptr addrspace(1)) "gc-leaf-function"

define void @test() gc "pylir-gc" {
    %a = alloca { ptr addrspace(1), [0 x ptr addrspace(1)] }, addrspace(1)
    call void @escape(ptr addrspace(1) %a)
    call void @builtins.__init__()
    call void @escape(ptr addrspace(1) %a)
    ret void
}

; CHECK-LABEL: define void @test()
; CHECK-NEXT: %[[ALLOCA:.*]] = alloca
; CHECK: %{{.*}} = call token ({{.*}}) @llvm.experimental.gc.statepoint{{.*}}({{.*}}@builtins.__init__{{.*}}) [ "deopt"(ptr addrspace(1) %[[ALLOCA]]) ]

define void @test2() gc "pylir-gc" {
    %a = alloca { ptr addrspace(1), [0 x ptr addrspace(1)] }, addrspace(1)
    call void @builtins.__init__()
    call void @llvm.lifetime.start.p1(i64 -1, ptr addrspace(1) %a)
    call void @escape(ptr addrspace(1) %a)
    ret void
}

; CHECK-LABEL: define void @test2()
; CHECK-NEXT: %[[ALLOCA:.*]] = alloca
; FIXME: This should be an empty deopt. The alloca is not yet live when 'builtins.__init__' gets called, and not even
;        initialized. Accessing it at that point in time would read undef.
; CHECK-NEXT: %{{.*}} = call token ({{.*}}) @llvm.experimental.gc.statepoint{{.*}}({{.*}}@builtins.__init__{{.*}}) [ "deopt"(ptr addrspace(1) %[[ALLOCA]]) ]