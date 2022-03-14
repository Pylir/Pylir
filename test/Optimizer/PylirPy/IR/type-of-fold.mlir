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

py.globalValue @builtins.type = #py.type<>
py.globalValue @a = #py.type<>

func @constant_obj() -> !py.dynamic {
    %0 = py.constant #py.obj<typeObject = @a>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @constant_obj
// CHECK: %[[CONST:.*]] = py.constant @a
// CHECK: return %[[CONST]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @a = #py.type<>

func @global_value() -> !py.dynamic {
    %0 = py.constant @a
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @global_value
// CHECK: %[[CONST:.*]] = py.constant @builtins.type
// CHECK: return %[[CONST]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.int = #py.type<>

func @int_attr() -> !py.dynamic {
    %0 = py.constant #py.int<value = 5>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @int_attr
// CHECK: %[[RESULT:.*]] = py.constant @builtins.int
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.float = #py.type<>

func @float_attr() -> !py.dynamic {
    %0 = py.constant #py.float<value = 5.0>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @float_attr
// CHECK: %[[RESULT:.*]] = py.constant @builtins.float
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.bool = #py.type<>

func @bool_attr() -> !py.dynamic {
    %0 = py.constant #py.bool<value = True>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @bool_attr
// CHECK: %[[RESULT:.*]] = py.constant @builtins.bool
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.tuple = #py.type<>

func @tuple_attr() -> !py.dynamic {
    %0 = py.constant #py.tuple<value = ()>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @tuple_attr
// CHECK: %[[RESULT:.*]] = py.constant @builtins.tuple
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.list = #py.type<>

func @list_attr() -> !py.dynamic {
    %0 = py.constant #py.list<value = []>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @list_attr
// CHECK: %[[RESULT:.*]] = py.constant @builtins.list
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.dict = #py.type<>

func @dict_attr() -> !py.dynamic {
    %0 = py.constant #py.dict<value = {}>
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @dict_attr
// CHECK: %[[RESULT:.*]] = py.constant @builtins.dict
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.tuple = #py.type<>

func @list_to_tuple(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.list.toTuple %arg0
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @list_to_tuple
// CHECK: %[[RESULT:.*]] = py.constant @builtins.tuple
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.tuple = #py.type<>

func @make_tuple() -> !py.dynamic {
    %0 = py.makeTuple ()
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @make_tuple
// CHECK: %[[RESULT:.*]] = py.constant @builtins.tuple
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.tuple = #py.type<>

func @prepend_tuple(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.tuple.prepend %arg0, %arg1
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @prepend_tuple
// CHECK: %[[RESULT:.*]] = py.constant @builtins.tuple
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.tuple = #py.type<>

func @tuple_pop_front(%arg0 : !py.dynamic) -> !py.dynamic {
    %element, %result = py.tuple.popFront %arg0
    %1 = py.typeOf %result
    return %1 : !py.dynamic
}

// CHECK-LABEL: @tuple_pop_front
// CHECK: %[[RESULT:.*]] = py.constant @builtins.tuple
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.tuple = #py.type<>
py.globalValue @builtins.BaseException = #py.type<>

func @make_tuple_ex(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTupleEx (*%arg0)
        label ^bb0 unwind ^bb1
^bb0:
    %1 = py.typeOf %0
    return %1 : !py.dynamic
^bb1:
    %2 = py.landingPad @builtins.BaseException
    py.landingPad.br ^bb2(%2)

^bb2(%e : !py.dynamic):
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

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.list = #py.type<>

func @make_list() -> !py.dynamic {
    %0 = py.makeList ()
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @make_list
// CHECK: %[[RESULT:.*]] = py.constant @builtins.list
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.list = #py.type<>
py.globalValue @builtins.BaseException = #py.type<>

func @make_list_ex(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeListEx (*%arg0)
        label ^bb0 unwind ^bb1
^bb0:
    %1 = py.typeOf %0
    return %1 : !py.dynamic
^bb1:
    %2 = py.landingPad @builtins.BaseException
    py.landingPad.br ^bb2(%2)

^bb2(%e : !py.dynamic):
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


py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.dict = #py.type<>

func @make_dict() -> !py.dynamic {
    %0 = py.makeDict ()
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @make_dict
// CHECK: %[[RESULT:.*]] = py.constant @builtins.dict
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.dict = #py.type<>
py.globalValue @builtins.BaseException = #py.type<>

func @make_dict_ex(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeDictEx (**%arg0)
        label ^bb0 unwind ^bb1
^bb0:
    %1 = py.typeOf %0
    return %1 : !py.dynamic
^bb1:
    %2 = py.landingPad @builtins.BaseException
    py.landingPad.br ^bb2(%2)

^bb2(%e : !py.dynamic):
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
    %0 = py.constant #py.int<value = 0>
    return %0 : !py.dynamic
}

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.None = #py.type<>
py.globalValue @builtins.function = #py.type<>
py.globalValue @builtins.int = #py.type<>

func @make_function() -> !py.dynamic {
    %0 = py.makeFunc @test
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @make_function
// CHECK: %[[RESULT:.*]] = py.constant @builtins.function
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.bool = #py.type<>

func @bool_from_I1(%arg0 : i1) -> !py.dynamic {
    %0 = py.bool.fromI1 %arg0
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @bool_from_I1
// CHECK: %[[RESULT:.*]] = py.constant @builtins.bool
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.str = #py.type<>

func @int_to_str(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.int.toStr %arg0
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @int_to_str
// CHECK: %[[RESULT:.*]] = py.constant @builtins.str
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.str = #py.type<>

func @str_concat(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    %0 = py.str.concat %arg0, %arg1, %arg2
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @str_concat
// CHECK: %[[RESULT:.*]] = py.constant @builtins.str
// CHECK: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.str = #py.type<>

func @str_copy(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.str.copy %arg0 : %arg1
    %1 = py.typeOf %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @str_copy
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: return %[[ARG1]]
