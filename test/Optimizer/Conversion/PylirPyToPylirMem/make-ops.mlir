// RUN: pylir-opt %s -convert-pylirPy-to-pylirMem --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

func @make_tuple(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_tuple
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = py.constant(@builtins.tuple)
// CHECK-NEXT: %[[SIZE:.*]] = arith.constant 1
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocTuple %[[TUPLE]][%[[SIZE]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initTuple %[[MEM]] to (%[[ARG]])
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.list = #py.type

func @make_list(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeList (%arg0)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_list
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[LIST:.*]] = py.constant(@builtins.list)
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[LIST]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initList %[[MEM]] to [%[[ARG]]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.set = #py.type

func @make_set(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeSet (%arg0)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_set
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[SET:.*]] = py.constant(@builtins.set)
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[SET]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initSet %[[MEM]] to {%[[ARG]]}
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.dict = #py.type

func @make_dict(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeDict (%arg0 : %arg1)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_dict
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[DICT:.*]] = py.constant(@builtins.dict)
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[DICT]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initDict %[[MEM]]
// CHECK-NEXT: py.dict.setItem %[[RESULT]][%[[ARG0]]] to %[[ARG1]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.function = #py.type

func private @test(!py.dynamic,!py.dynamic,!py.dynamic) -> !py.dynamic

func @make_function() -> !py.dynamic {
    %0 = py.makeFunc @test
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_function
// CHECK-NEXT: %[[FUNCTION:.*]] = py.constant(@builtins.function)
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[FUNCTION]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initFunc %[[MEM]] to @test
// CHECK-NEXT: return %[[RESULT]]

// -----

func @make_object(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeObject %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_object
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[ARG]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initObject %[[MEM]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

func @make_tuple_from_list(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.list.toTuple %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_tuple_from_list
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = py.constant(@builtins.tuple)
// CHECK-NEXT: %[[SIZE:.*]] = py.list.len %[[ARG]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocTuple %[[TUPLE]][%[[SIZE]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initTupleFromList %[[MEM]] to (* %[[ARG]])
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type

func @make_bool_from_i1(%arg0 : i1) -> !py.dynamic {
    %0 = py.bool.fromI1 %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_bool_from_i1
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[BOOL:.*]] = py.constant(@builtins.bool)
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[BOOL]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initInt %[[MEM]] to %[[ARG]] : i1
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @make_int_fromInteger(%arg0 : i32) -> !py.dynamic {
    %0 = py.int.fromInteger %arg0 : i32
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_int_fromInteger
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[BOOL:.*]] = py.constant(@builtins.int)
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[BOOL]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initInt %[[MEM]] to %[[ARG]] : i32
// CHECK-NEXT: return %[[RESULT]]


// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.str = #py.type

func @make_str_fromInt(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.int.toStr %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_str_fromInt
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[STR:.*]] = py.constant(@builtins.str)
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[STR]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initStrFromInt %[[MEM]] to %[[ARG]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @make_int_from_add(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.int.add %arg0, %arg1
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_int_from_add
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[INT:.*]] = py.constant(@builtins.int)
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[INT]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initIntAdd %[[MEM]] to %[[ARG0]] + %[[ARG1]]
// CHECK-NEXT: return %[[RESULT]]
