// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, const, initializer = #py.type<instance_slots = #py.tuple<(#py.str<"__eq__">,#py.str<"__hash__">)>, mro_tuple = #py.tuple<(#py.globalValue<builtins.type>)>>>
py.external @builtins.type, #builtins_type

#builtins_object = #py.globalValue<builtins.object, const, initializer = #py.type>
py.external @builtins.object, #builtins_object
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

// CHECK-LABEL: llvm.mlir.global external constant @builtins.type
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.type
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[TYPE]], %[[UNDEF]][0]
// CHECK-NEXT: %[[AS_COUNT:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[AS_COUNT]], %[[UNDEF1]][1]
// CHECK-NEXT: %[[LAYOUT:.*]] = llvm.mlir.addressof @builtins.type
// CHECK-NEXT: %[[UNDEF3:.*]] = llvm.insertvalue %[[LAYOUT]], %[[UNDEF2]][2]
// CHECK-NEXT: %[[MRO:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[UNDEF4:.*]] = llvm.insertvalue %[[MRO]], %[[UNDEF3]][3]
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[UNDEF5:.*]] = llvm.insertvalue %[[ADDRESS]], %[[UNDEF4]][4]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.zero
// CHECK-NEXT: %[[UNDEF6:.*]] = llvm.insertvalue %[[NULL]], %[[UNDEF5]][5, 0]
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.zero
// CHECK-NEXT: %[[UNDEF7:.*]] = llvm.insertvalue %[[NULL]], %[[UNDEF6]][5, 1]
// CHECK-NEXT: llvm.return %[[UNDEF7]]
