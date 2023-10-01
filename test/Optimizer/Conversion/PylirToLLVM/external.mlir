// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
#sys = #py.globalValue<sys>

// CHECK-LABEL: llvm.mlir.global external @builtins.int{{.*\{$}}
// CHECK: llvm.return
py.external @builtins.int, #builtins_int

// CHECK-LABEL: llvm.mlir.global external @builtins.sys()
// Check that this instance ends with the type rather than also having an initializer region after.
// CHECK-SAME: : !llvm.struct<{{.*}}>{{\s*$}}
py.external @builtins.sys, #sys
