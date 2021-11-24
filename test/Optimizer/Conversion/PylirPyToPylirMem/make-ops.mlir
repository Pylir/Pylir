// RUN: pylir-opt %s -convert-pylirPy-to-pylirMem --split-input-file | FileCheck %s

func @make_tuple(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_tuple
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = py.constant @builtins.tuple
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[TUPLE]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initTuple %[[MEM]] to (%[[ARG]])
// CHECK-NEXT: return %[[RESULT]]

// -----

func @make_list(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeList (%arg0)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_list
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[LIST:.*]] = py.constant @builtins.list
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[LIST]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initList %[[MEM]] to [%[[ARG]]]
// CHECK-NEXT: return %[[RESULT]]

// -----

func @make_set(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeSet (%arg0)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_set
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[SET:.*]] = py.constant @builtins.set
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[SET]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initSet %[[MEM]] to {%[[ARG]]}
// CHECK-NEXT: return %[[RESULT]]

// -----

func @make_dict(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeDict (%arg0 : %arg1)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_dict
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[DICT:.*]] = py.constant @builtins.dict
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[DICT]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initDict %[[MEM]]
// CHECK-NEXT: py.dict.setItem %[[RESULT]][%[[ARG0]]] to %[[ARG1]]
// CHECK-NEXT: return %[[RESULT]]

// -----

func private @test(!py.dynamic,!py.dynamic,!py.dynamic) -> !py.dynamic

func @make_function() -> !py.dynamic {
    %0 = py.makeFunc @test
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_function
// CHECK-NEXT: %[[FUNCTION:.*]] = py.constant @builtins.function
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

func @make_tuple_from_list(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.list.toTuple %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_tuple
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = py.constant @builtins.tuple
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[TUPLE]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initTupleFromList %[[MEM]] to (* %[[ARG]])
// CHECK-NEXT: return %[[RESULT]]

// -----

func @make_tuple_from_list(%arg0 : i1) -> !py.dynamic {
    %0 = py.bool.fromI1 %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_tuple
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[BOOL:.*]] = py.constant @builtins.bool
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[BOOL]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initInt %[[MEM]] to %[[ARG]] : i1
// CHECK-NEXT: return %[[RESULT]]
