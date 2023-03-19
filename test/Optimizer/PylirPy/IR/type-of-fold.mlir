// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func @make_object(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeObject %arg0
    %1 = typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @make_object
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: return %[[ARG0]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @a = #py.type

py.func @constant_obj() -> !py.dynamic {
    %0 = constant(#py.obj<#py.ref<@a>>)
    %1 = typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @constant_obj
// CHECK: %[[CONST:.*]] = constant(#py.ref<@a>)
// CHECK: return %[[CONST]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @a = #py.type

py.func @global_value() -> !py.dynamic {
    %0 = constant(#py.ref<@a>)
    %1 = typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @global_value
// CHECK: %[[CONST:.*]] = constant(#py.ref<@builtins.type>)
// CHECK: return %[[CONST]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.func @str_copy(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.str.copy %arg0 : %arg1
    %1 = typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @str_copy
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: return %[[ARG1]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

py.func @type_refineable(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0, %arg1)
    %1 = typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @type_refineable
// CHECK: %[[CONST:.*]] = constant(#py.ref<@builtins.tuple>)
// CHECK: return %[[CONST]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

py.func @tuple_prepend(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.tuple.prepend %arg0, %arg1
    %1 = typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @tuple_prepend
// CHECK: %[[CONST:.*]] = constant(#py.ref<@builtins.tuple>)
// CHECK: return %[[CONST]]
