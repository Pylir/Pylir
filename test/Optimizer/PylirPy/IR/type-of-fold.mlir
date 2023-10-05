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

#a = #py.globalValue<a, initializer = #py.type>

py.func @constant_obj() -> !py.dynamic {
    %0 = constant(#py.obj<#a>)
    %1 = typeOf %0
    return %1 : !py.dynamic
}

// CHECK: #[[$A:.*]] = #py.globalValue<a{{.*}}>

// CHECK-LABEL: @constant_obj
// CHECK: %[[CONST:.*]] = constant(#[[$A]])
// CHECK: return %[[CONST]]

// -----

#a = #py.globalValue<a, initializer = #py.type>

py.func @global_value() -> !py.dynamic {
    %0 = constant(#a)
    %1 = typeOf %0
    return %1 : !py.dynamic
}

// CHECK: #[[$TYPE:.*]] = #py.globalValue<builtins.type{{.*}}>

// CHECK-LABEL: @global_value
// CHECK: %[[CONST:.*]] = constant(#[[$TYPE]])
// CHECK: return %[[CONST]]

// -----

py.func @str_copy(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = str_copy %arg0 : %arg1
    %1 = typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @str_copy
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: return %[[ARG1]]

// -----

py.func @type_refineable(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0, %arg1)
    %1 = typeOf %0
    return %1 : !py.dynamic
}

// CHECK: #[[$TUPLE:.*]] = #py.globalValue<builtins.tuple{{.*}}>

// CHECK-LABEL: @type_refineable
// CHECK: %[[CONST:.*]] = constant(#[[$TUPLE]])
// CHECK: return %[[CONST]]

// -----

py.func @tuple_prepend(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = tuple_prepend %arg0, %arg1
    %1 = typeOf %0
    return %1 : !py.dynamic
}

// CHECK: #[[$TUPLE:.*]] = #py.globalValue<builtins.tuple{{.*}}>

// CHECK-LABEL: @tuple_prepend
// CHECK: %[[CONST:.*]] = constant(#[[$TUPLE]])
// CHECK: return %[[CONST]]

// -----

// CHECK: #[[$U:.*]] = #py.globalValue<imported>

// CHECK-LABEL: func @unknown_type
py.func @unknown_type() -> !py.dynamic {
    // CHECK: %[[C:.*]] = constant(#[[$U]])
    // CHECK-NEXT: return %[[C]]

    %0 = constant(#py.obj<#py.globalValue<imported>>)
    %1 = typeOf %0
    return %1 : !py.dynamic
}
