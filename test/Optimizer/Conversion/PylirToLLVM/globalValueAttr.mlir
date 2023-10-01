// RUN: pylir-opt %s -convert-pylir-to-llvm='target-triple=x86_64-unknown-linux-gnu' --split-input-file | FileCheck %s --check-prefixes=CHECK,SECTION_COFF
// RUN: pylir-opt %s -convert-pylir-to-llvm='target-triple=x86_64-apple-darwin21.6.0' --split-input-file | FileCheck %s --check-prefixes=CHECK,SECTION_MACHO

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>

// CHECK-LABEL: func @test()
py.func @test() -> !py.dynamic {
    // CHECK: llvm.mlir.addressof @[[$SYMBOL:.*]] : !llvm.ptr
    %0 = py.constant(#builtins_int)
    return %0 : !py.dynamic
}

// CHECK: llvm.mlir.global internal @[[$SYMBOL]]()

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_object = #py.globalValue<builtins.object, initializer = #py.type>
py.external @builtins.object, #builtins_object
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#foo = #py.globalValue<foo, const, initializer = #py.tuple<()>>
py.external @foo, #foo
#bar = #py.globalValue<bar, const, initializer = #py.tuple<(#foo, #builtins_type)>>
py.external @bar, #bar

// CHECK-LABEL: llvm.mlir.global external constant @foo
// SECTION_COFF-SAME: section = "py_const"
// SECTION_MACHO-SAME: section = "__DATA,py_const"
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[TYPE]], %[[UNDEF]][0]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(0 : i{{.*}})
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1]
// CHECK-NEXT: llvm.return %[[UNDEF2]]

// CHECK-LABEL: llvm.mlir.global external constant @bar
// SECTION_COFF-SAME: section = "py_const"
// SECTION_MACHO-SAME: section = "__DATA,py_const"
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[TYPE]], %[[UNDEF]][0]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(2 : i{{.*}})
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1]
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.mlir.addressof @foo
// CHECK-NEXT: %[[UNDEF3:.*]] = llvm.insertvalue %[[ADDRESS]], %[[UNDEF2]][2, 0]
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.mlir.addressof @builtins.type
// CHECK-NEXT: %[[UNDEF4:.*]] = llvm.insertvalue %[[ADDRESS]], %[[UNDEF3]][2, 1]
// CHECK-NEXT: llvm.return %[[UNDEF4]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

#foo = #py.globalValue<foo, const, initializer = #py.str<"test">>
py.external @foo, #foo

// CHECK: llvm.mlir.global private unnamed_addr constant @[[$BUFFER:buffer\$[[:alnum:]]*]]("test")

// CHECK-LABEL: llvm.mlir.global external constant @foo
// SECTION_COFF-SAME: section = "py_const"
// SECTION_MACHO-SAME: section = "__DATA,py_const"
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.str
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[TYPE]], %[[UNDEF]][0]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant(4 : i64)
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF1]][1, 0]
// CHECK-NEXT: %[[UNDEF3:.*]] = llvm.insertvalue %[[SIZE]], %[[UNDEF2]][1, 1]
// CHECK-NEXT: %[[BUFFER_ADDR:.*]] = llvm.mlir.addressof @[[$BUFFER]]
// CHECK-NEXT: %[[UNDEF4:.*]] = llvm.insertvalue %[[BUFFER_ADDR]], %[[UNDEF3]][1, 2]
// CHECK-NEXT: llvm.return %[[UNDEF4]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_list = #py.globalValue<builtins.list, initializer = #py.type>

#bar = #py.globalValue<bar, initializer = #py.list<[]>>
py.external @bar, #bar
#foo = #py.globalValue<foo, const, initializer = #py.list<[]>>
py.external @foo, #foo

// CHECK: llvm.mlir.global external @bar
// SECTION_COFF-SAME: section = "py_coll"
// SECTION_MACHO-SAME: section = "__DATA,py_coll"
// CHECK: llvm.mlir.global external constant @foo
// SECTION_COFF-SAME: section = "py_const"
// SECTION_MACHO-SAME: section = "__DATA,py_const"

// -----

#v = #py.globalValue<v, const>
py.external @sys.__excepthook__, #v

// CHECK: llvm.mlir.global external constant @sys.__excepthook__()

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_float = #py.globalValue<builtins.float, initializer = #py.type>
py.external @builtins.float, #builtins_float

#bar = #py.globalValue<bar, const, initializer = #py.float<5.25>>
py.external @bar, #bar

// CHECK-LABEL: llvm.mlir.global external constant @bar
// CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.float
// CHECK-NEXT: %[[UNDEF1:.*]] = llvm.insertvalue %[[TYPE]], %[[UNDEF]][0]
// CHECK-NEXT: %[[VALUE:.*]] = llvm.mlir.constant(5.25{{0*}}e+{{0+}} : f64)
// CHECK-NEXT: %[[UNDEF2:.*]] = llvm.insertvalue %[[VALUE]], %[[UNDEF1]][1]
// CHECK-NEXT: llvm.return %[[UNDEF2]]
