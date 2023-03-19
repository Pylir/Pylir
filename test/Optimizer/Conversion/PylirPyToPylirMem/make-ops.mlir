// RUN: pylir-opt %s -convert-pylirPy-to-pylirMem --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

py.func @make_tuple(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_tuple
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = constant(#py.ref<@builtins.tuple>)
// CHECK-NEXT: %[[SIZE:.*]] = arith.constant 1
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[TUPLE]][%[[SIZE]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initTuple %[[MEM]] to (%[[ARG]])
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.list = #py.type

py.func @make_list(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeList (%arg0)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_list
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[LIST:.*]] = constant(#py.ref<@builtins.list>)
// CHECK-NEXT: %[[SLOTS:.*]] = py.type.slots %[[LIST]]
// CHECK-NEXT: %[[LEN:.*]] = py.tuple.len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[LIST]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initList %[[MEM]] to [%[[ARG]]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.set = #py.type

py.func @make_set(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeSet (%arg0)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_set
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[SET:.*]] = constant(#py.ref<@builtins.set>)
// CHECK-NEXT: %[[SLOTS:.*]] = py.type.slots %[[SET]]
// CHECK-NEXT: %[[LEN:.*]] = py.tuple.len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[SET]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initSet %[[MEM]] to {%[[ARG]]}
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.dict = #py.type

py.func @make_dict(%arg0 : !py.dynamic, %arg1: index, %arg2 : !py.dynamic) -> !py.dynamic {
    %0 = makeDict (%arg0 hash(%arg1) : %arg2)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_dict
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-NEXT: %[[DICT:.*]] = constant(#py.ref<@builtins.dict>)
// CHECK-NEXT: %[[SLOTS:.*]] = py.type.slots %[[DICT]]
// CHECK-NEXT: %[[LEN:.*]] = py.tuple.len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[DICT]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initDict %[[MEM]]
// CHECK-NEXT: py.dict.setItem %[[RESULT]][%[[ARG0]] hash(%[[ARG1]])] to %[[ARG2]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.function = #py.type

py.func private @test(!py.dynamic,!py.dynamic,!py.dynamic) -> !py.dynamic

py.func @make_function() -> !py.dynamic {
    %0 = makeFunc @test
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_function
// CHECK-NEXT: %[[FUNCTION:.*]] = constant(#py.ref<@builtins.function>)
// CHECK-NEXT: %[[SLOTS:.*]] = py.type.slots %[[FUNCTION]]
// CHECK-NEXT: %[[LEN:.*]] = py.tuple.len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[FUNCTION]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initFunc %[[MEM]] to @test
// CHECK-NEXT: return %[[RESULT]]

// -----

py.func @make_object(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeObject %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_object
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[SLOTS:.*]] = py.type.slots %[[ARG]]
// CHECK-NEXT: %[[LEN:.*]] = py.tuple.len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[ARG]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initObject %[[MEM]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

py.func @make_tuple_from_list(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.list.toTuple %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_tuple_from_list
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = constant(#py.ref<@builtins.tuple>)
// CHECK-NEXT: %[[SIZE:.*]] = py.list.len %[[ARG]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[TUPLE]][%[[SIZE]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initTupleFromList %[[MEM]] to (* %[[ARG]])
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.bool = #py.type

py.func @make_bool_from_i1(%arg0 : i1) -> !py.dynamic {
    %0 = py.bool.fromI1 %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_bool_from_i1
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[EXT:.*]] = arith.extui %[[ARG]] : i1 to i{{[0-9]+}}
// CHECK-NEXT: %[[INDEX:.*]] = arith.index_cast %[[EXT]]
// CHECK-NEXT: %[[BOOL:.*]] = constant(#py.ref<@builtins.bool>)
// CHECK-NEXT: %[[SLOTS:.*]] = py.type.slots %[[BOOL]]
// CHECK-NEXT: %[[LEN:.*]] = py.tuple.len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[BOOL]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initIntUnsigned %[[MEM]] to %[[INDEX]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

py.func @make_int_fromInteger(%arg0 : index) -> !py.dynamic {
    %0 = py.int.fromUnsigned %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_int_fromInteger
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[BOOL:.*]] = constant(#py.ref<@builtins.int>)
// CHECK-NEXT: %[[SLOTS:.*]] = py.type.slots %[[BOOL]]
// CHECK-NEXT: %[[LEN:.*]] = py.tuple.len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[BOOL]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initIntUnsigned %[[MEM]] to %[[ARG]]
// CHECK-NEXT: return %[[RESULT]]


// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.str = #py.type

py.func @make_str_fromInt(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.int.toStr %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_str_fromInt
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[STR:.*]] = constant(#py.ref<@builtins.str>)
// CHECK-NEXT: %[[SLOTS:.*]] = py.type.slots %[[STR]]
// CHECK-NEXT: %[[LEN:.*]] = py.tuple.len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[STR]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initStrFromInt %[[MEM]] to %[[ARG]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

py.func @make_int_from_add(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.int.add %arg0, %arg1
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_int_from_add
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[INT:.*]] = constant(#py.ref<@builtins.int>)
// CHECK-NEXT: %[[SLOTS:.*]] = py.type.slots %[[INT]]
// CHECK-NEXT: %[[LEN:.*]] = py.tuple.len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[INT]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initIntAdd %[[MEM]] to %[[ARG0]] + %[[ARG1]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.float = #py.type

py.func @make_float_fromF64(%arg0 : f64) -> !py.dynamic {
    %0 = py.float.fromF64 %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_float_fromF64
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[FLOAT:.*]] = constant(#py.ref<@builtins.float>)
// CHECK-NEXT: %[[SLOTS:.*]] = py.type.slots %[[FLOAT]]
// CHECK-NEXT: %[[LEN:.*]] = py.tuple.len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[FLOAT]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initFloat %[[MEM]] to %[[ARG]]
// CHECK-NEXT: return %[[RESULT]]
