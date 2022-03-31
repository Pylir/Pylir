// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func @make_object(%arg0 : !py.unknown) -> !py.unknown {
    %0 = py.makeObject %arg0 : (!py.unknown) -> !py.unknown
    %1 = py.typeOf %0 : (!py.unknown) -> !py.unknown
    return %1 : !py.unknown
}

// CHECK-LABEL: @make_object
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: return %[[ARG0]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @a = #py.type

func @constant_obj() -> !py.unknown {
    %0 = py.constant(#py.obj<@a>) : !py.unknown
    %1 = py.typeOf %0 : (!py.unknown) -> !py.unknown
    return %1 : !py.unknown
}

// CHECK-LABEL: @constant_obj
// CHECK: %[[CONST:.*]] = py.constant(@a)
// CHECK: return %[[CONST]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @a = #py.type

func @global_value() -> !py.unknown {
    %0 = py.constant(@a) : !py.unknown
    %1 = py.typeOf %0 : (!py.unknown) -> !py.unknown
    return %1 : !py.unknown
}

// CHECK-LABEL: @global_value
// CHECK: %[[CONST:.*]] = py.constant(@builtins.type)
// CHECK: return %[[CONST]]

// -----

py.globalValue @builtins.type = #py.type

func @typed(%arg0 : !py.class<@builtins.type>) -> !py.unknown {
    %0 = py.typeOf %arg0 : (!py.class<@builtins.type>) -> !py.unknown
    return %0 : !py.unknown
}

// CHECK-LABEL: @typed
// CHECK: %[[CONST:.*]] = py.constant(@builtins.type)
// CHECK: return %[[CONST]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func @str_copy(%arg0 : !py.unknown, %arg1 : !py.unknown) -> !py.unknown {
    %0 = py.str.copy %arg0 : %arg1 : (!py.unknown, !py.unknown) -> !py.unknown
    %1 = py.typeOf %0 : (!py.unknown) -> !py.unknown
    return %1 : !py.unknown
}

// CHECK-LABEL: @str_copy
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: return %[[ARG1]]
