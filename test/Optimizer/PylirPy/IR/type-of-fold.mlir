// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func @make_object(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeObject %arg0
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @make_object
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: return %[[ARG0]]

// -----

py.globalValue @a = #py.int<0>

func @constant_obj() -> !py.dynamic {
    %0 = py.constant #py.obj<type: @type, __dict__: #py.dict<{}>>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @constant_obj
// CHECK: %[[CONST:.*]] = py.constant @type
// CHECK: return %[[CONST]]

// -----

py.globalValue @builtins.int = #py.int<0>

func @int_attr() -> !py.dynamic {
    %0 = py.constant #py.int<5>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @int_attr
// CHECK: %[[RESULT:.*]] = py.constant @builtins.int
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.float = #py.int<0>

func @float_attr() -> !py.dynamic {
    %0 = py.constant #py.float<5.0>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @float_attr
// CHECK: %[[RESULT:.*]] = py.constant @builtins.float
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.bool = #py.int<0>

func @bool_attr() -> !py.dynamic {
    %0 = py.constant #py.bool<True>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @bool_attr
// CHECK: %[[RESULT:.*]] = py.constant @builtins.bool
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.tuple = #py.int<0>

func @tuple_attr() -> !py.dynamic {
    %0 = py.constant #py.tuple<()>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @tuple_attr
// CHECK: %[[RESULT:.*]] = py.constant @builtins.tuple
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.list = #py.int<0>

func @list_attr() -> !py.dynamic {
    %0 = py.constant #py.list<[]>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @list_attr
// CHECK: %[[RESULT:.*]] = py.constant @builtins.list
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.dict = #py.int<0>

func @dict_attr() -> !py.dynamic {
    %0 = py.constant #py.dict<{}>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @dict_attr
// CHECK: %[[RESULT:.*]] = py.constant @builtins.dict
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.tuple = #py.int<0>

func @list_to_tuple(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.list.toTuple %arg0
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @list_to_tuple
// CHECK: %[[RESULT:.*]] = py.constant @builtins.tuple
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.tuple = #py.int<0>

func @make_tuple() -> !py.dynamic {
    %0 = py.makeTuple ()
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @make_tuple
// CHECK: %[[RESULT:.*]] = py.constant @builtins.tuple
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.tuple = #py.int<0>

func @make_tuple_ex(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTupleEx (*%arg0)
        label ^bb0 unwind ^bb1
^bb0:
    %1 = py.typeOf %0
    return %1 : !py.dynamic
^bb1(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_tuple_ex
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[RESULT:.*]] = py.constant @builtins.tuple
// CHECK: py.makeTupleEx (*%[[ARG0]])
// CHECK-NEXT: label ^[[HAPPY:[[:alnum:]]+]]

// CHECK: ^[[HAPPY]]:
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.list = #py.int<0>

func @make_list() -> !py.dynamic {
    %0 = py.makeList ()
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @make_list
// CHECK: %[[RESULT:.*]] = py.constant @builtins.list
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.list = #py.int<0>

func @make_list_ex(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeListEx (*%arg0)
        label ^bb0 unwind ^bb1
^bb0:
    %1 = py.typeOf %0
    return %1 : !py.dynamic
^bb1(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_list_ex
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[RESULT:.*]] = py.constant @builtins.list
// CHECK: py.makeListEx (*%[[ARG0]])
// CHECK-NEXT: label ^[[HAPPY:[[:alnum:]]+]]

// CHECK: ^[[HAPPY]]:
// CHECK: return %[[RESULT]]

// -----


py.globalValue @builtins.dict = #py.int<0>

func @make_dict() -> !py.dynamic {
    %0 = py.makeDict ()
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @make_dict
// CHECK: %[[RESULT:.*]] = py.constant @builtins.dict
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.dict = #py.int<0>

func @make_dict_ex(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeDictEx (**%arg0)
        label ^bb0 unwind ^bb1
^bb0:
    %1 = py.typeOf %0
    return %1 : !py.dynamic
^bb1(%e : !py.dynamic):
    return %e : !py.dynamic
}

// CHECK-LABEL: @make_dict_ex
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[RESULT:.*]] = py.constant @builtins.dict
// CHECK: py.makeDictEx (**%[[ARG0]])
// CHECK-NEXT: label ^[[HAPPY:[[:alnum:]]+]]

// CHECK: ^[[HAPPY]]:
// CHECK: return %[[RESULT]]

// -----

func private @test(%self : !py.dynamic, %tuple : !py.dynamic, %dict : !py.dynamic) -> !py.dynamic {
    %0 = py.constant #py.int<0>
    return %0 : !py.dynamic
}

py.globalValue @builtins.function = #py.int<0>

func @make_function() -> !py.dynamic {
    %0 = py.makeFunc @test
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @make_function
// CHECK: %[[RESULT:.*]] = py.constant @builtins.function
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.bool = #py.int<0>

func @bool_from_I1(%arg0 : i1) -> !py.dynamic {
    %0 = py.bool.fromI1 %arg0
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @bool_from_I1
// CHECK: %[[RESULT:.*]] = py.constant @builtins.bool
// CHECK: return %[[RESULT]]

// -----

func private @test(%self : !py.dynamic, %tuple : !py.dynamic, %dict : !py.dynamic) -> !py.dynamic {
    %0 = py.constant #py.int<0>
    return %0 : !py.dynamic
}

py.globalValue @builtins.function = #py.int<0>

func @get_function(%arg0 : !py.dynamic) -> !py.dynamic {
    %0, %1 = py.getFunction %arg0
    %2 = py.typeOf %0
    return %2 : !py.dynamic
}

// CHECK-LABEL: @get_function
// CHECK: %[[RESULT:.*]] = py.constant @builtins.function
// CHECK: return %[[RESULT]]
