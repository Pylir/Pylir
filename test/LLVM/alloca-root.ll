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
; CHECK-NEXT: store ptr addrspace(1) null, ptr addrspace(1) %[[ALLOCA]]
; CHECK-NEXT: %{{.*}} = call token ({{.*}}) @llvm.experimental.gc.statepoint{{.*}}({{.*}}@builtins.__init__{{.*}}) [ "deopt"() ]

declare i1 @random() "gc-leaf-function"

define void @test3() gc "pylir-gc" {
    %a = alloca { ptr addrspace(1), [0 x ptr addrspace(1)] }, addrspace(1)
    call void @builtins.__init__()
    %c = call i1 @random()
    br i1 %c, label %true, label %false

true:
    call void @llvm.lifetime.start.p1(i64 -1, ptr addrspace(1) %a)
    call void @escape(ptr addrspace(1) %a)
    br label %merge

false:
    br label %merge

merge:
    ret void
}

; CHECK-LABEL: define void @test3()
; CHECK-NEXT: %[[ALLOCA:.*]] = alloca
; CHECK-NEXT: store ptr addrspace(1) null, ptr addrspace(1) %[[ALLOCA]]
; CHECK-NEXT: call token ({{.*}}) @llvm.experimental.gc.statepoint{{.*}}({{.*}}@builtins.__init__{{.*}}) [ "deopt"() ]

define void @test4() gc "pylir-gc" {
    %a = alloca { ptr addrspace(1), [0 x ptr addrspace(1)] }, addrspace(1)
    call void @builtins.__init__()
    %c = call i1 @random()
    br i1 %c, label %true, label %false

true:
    call void @escape(ptr addrspace(1) %a)
    br label %merge

false:
    br label %merge

merge:
    ret void
}

; CHECK-LABEL: define void @test4()
; CHECK-NEXT: %[[ALLOCA:.*]] = alloca
; CHECK-NEXT: store ptr addrspace(1) null, ptr addrspace(1) %[[ALLOCA]]
; CHECK-NEXT: call token ({{.*}}) @llvm.experimental.gc.statepoint{{.*}}({{.*}}@builtins.__init__{{.*}}) [ "deopt"(ptr addrspace(1) %[[ALLOCA]]) ]

define void @test5() gc "pylir-gc" {
    %a = alloca { ptr addrspace(1), [0 x ptr addrspace(1)] }, addrspace(1)
    %c = call i1 @random()
    br i1 %c, label %bb0, label %false

bb0:
    call void @escape(ptr addrspace(1) %a)
    call void @llvm.lifetime.start.p1(i64 -1, ptr addrspace(1) %a)
    call void @builtins.__init__()
    call void @escape(ptr addrspace(1) %a)
    ret void

false:
    ret void
}

; CHECK-LABEL: define void @test5()
; CHECK-NEXT: %[[ALLOCA:.*]] = alloca
; CHECK-NEXT: store ptr addrspace(1) null, ptr addrspace(1) %[[ALLOCA]]
; CHECK: call token ({{.*}}) @llvm.experimental.gc.statepoint{{.*}}({{.*}}@builtins.__init__{{.*}}) [ "deopt"(ptr addrspace(1) %[[ALLOCA]]) ]
