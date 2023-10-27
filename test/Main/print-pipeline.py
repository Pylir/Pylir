# REQUIRES: x86-registered-target

# RUN: pylir %s -o %t --target=x86_64-unknown-linux-gnu -c -Xprint-pipeline 2>&1 | FileCheck %s
# CHECK: convert-pylir-to-llvm{data-layout=e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128 target-triple=x86_64-unknown-linux-gnu}
