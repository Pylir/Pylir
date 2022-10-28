; RUN: mkdir -p %t
; RUN: pylir %s -emit-llvm -o %t/dependency-file.bc -M - | FileCheck %s

; CHECK: dependency-file.bc:{{[[:space:]]*$}}