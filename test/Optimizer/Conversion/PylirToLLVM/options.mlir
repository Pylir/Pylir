// RUN: pylir-opt %s -convert-pylir-to-llvm='target-triple=x86_64-unknown-linux-gnu' | FileCheck %s --check-prefixes=TRIPLE,CHECK
// RUN: pylir-opt %s -convert-pylir-to-llvm='data-layout=e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128' | FileCheck %s --check-prefixes=DATALAYOUT,CHECK

// CHECK: module attributes
// TRIPLE-SAME: llvm.target_triple = "x86_64-unknown-linux-gnu"
// DATALAYOUT-SAME: llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
