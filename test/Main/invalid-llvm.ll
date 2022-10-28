; RUN: not pylir -emit-llvm -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=SYNTAX
; RUN: not pylir -emit-llvm -o /dev/null %t/nonExistent.ll 2>&1 | FileCheck %s --check-prefix=IO

#

; SYNTAX: error: expected top-level entity
; IO: error: Could not open input file: