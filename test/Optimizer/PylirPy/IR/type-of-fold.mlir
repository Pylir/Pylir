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

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
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

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
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

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

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

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

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

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @tuple_prepend(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = tuple_prepend %arg0, %arg1
    %1 = typeOf %0
    return %1 : !py.dynamic
}

// CHECK: #[[$TUPLE:.*]] = #py.globalValue<builtins.tuple{{.*}}>

// CHECK-LABEL: @tuple_prepend
// CHECK: %[[CONST:.*]] = constant(#[[$TUPLE]])
// CHECK: return %[[CONST]]
