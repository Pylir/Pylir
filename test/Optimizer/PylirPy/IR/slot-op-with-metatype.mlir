// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue const @const$ = #py.tuple<(#py.str<"__slots__">)>
py.globalValue @builtins.type = #py.type<slots = {__slots__ = @const$}>
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.typeOf %arg0
    %1 = py.typeOf %0
    %2 = py.getSlot "__slots__" from %0 : %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: func @foo
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[META_TYPE:.*]] = py.constant(@builtins.type)
// CHECK-NEXT: %[[TYPE:.*]] = py.typeOf %[[ARG0]]
// CHECK-NEXT: %[[RESULT:.*]] = py.getSlot "__slots__" from %[[TYPE]] : %[[META_TYPE]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue const @const$ = #py.tuple<(#py.str<"__slots__">)>
py.globalValue @builtins.type = #py.type<slots = {__slots__ = @const$}>
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic) {
    %0 = py.typeOf %arg0
    %1 = py.typeOf %0
    py.setSlot "__slots__" of %0 : %1 to %arg1
    return
}

// CHECK-LABEL: func @foo
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[META_TYPE:.*]] = py.constant(@builtins.type)
// CHECK-NEXT: %[[TYPE:.*]] = py.typeOf %[[ARG0]]
// CHECK-NEXT: py.setSlot "__slots__" of %[[TYPE]] : %[[META_TYPE]] to %[[ARG1]]
