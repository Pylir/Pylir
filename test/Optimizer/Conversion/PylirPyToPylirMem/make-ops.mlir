// RUN: pylir-opt %s -convert-pylirPy-to-pylirMem --split-input-file | FileCheck %s

// CHECK: #[[$TUPLE_TYPE:.*]] = #py.globalValue<builtins.tuple{{,|>}}

py.func @make_tuple(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_tuple
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = constant(#[[$TUPLE_TYPE]])
// CHECK-NEXT: %[[SIZE:.*]] = arith.constant 1
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[TUPLE]][%[[SIZE]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initTuple %[[MEM]] to (%[[ARG]])
// CHECK-NEXT: return %[[RESULT]]

// -----

// CHECK: #[[$LIST_TYPE:.*]] = #py.globalValue<builtins.list{{,|>}}

py.func @make_list(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeList (%arg0)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_list
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[LIST:.*]] = constant(#[[$LIST_TYPE]])
// CHECK-NEXT: %[[SLOTS:.*]] = type_slots %[[LIST]]
// CHECK-NEXT: %[[LEN:.*]] = tuple_len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[LIST]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initList %[[MEM]] to [%[[ARG]]]
// CHECK-NEXT: return %[[RESULT]]

// -----

// CHECK: #[[$SET_TYPE:.*]] = #py.globalValue<builtins.set{{,|>}}

py.func @make_set(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeSet (%arg0)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_set
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[SET:.*]] = constant(#[[$SET_TYPE]])
// CHECK-NEXT: %[[SLOTS:.*]] = type_slots %[[SET]]
// CHECK-NEXT: %[[LEN:.*]] = tuple_len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[SET]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initSet %[[MEM]] to {%[[ARG]]}
// CHECK-NEXT: return %[[RESULT]]

// -----

// CHECK: #[[$DICT_TYPE:.*]] = #py.globalValue<builtins.dict{{,|>}}

py.func @make_dict(%arg0 : !py.dynamic, %arg1: index, %arg2 : !py.dynamic) -> !py.dynamic {
    %0 = makeDict (%arg0 hash(%arg1) : %arg2)
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_dict
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-NEXT: %[[DICT:.*]] = constant(#[[$DICT_TYPE]])
// CHECK-NEXT: %[[SLOTS:.*]] = type_slots %[[DICT]]
// CHECK-NEXT: %[[LEN:.*]] = tuple_len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[DICT]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initDict %[[MEM]]
// CHECK-NEXT: dict_setItem %[[RESULT]][%[[ARG0]] hash(%[[ARG1]])] to %[[ARG2]]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.func private @test(!py.dynamic,!py.dynamic,!py.dynamic) -> !py.dynamic

py.func @make_function(%arg0 : i32, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = makeFunc @test [%arg0, %arg1 : i32, !py.dynamic]
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_function
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocFunction [i32, !py.dynamic]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initFunc %[[MEM]] to @test
// CHECK-SAME: [%[[ARG0]], %[[ARG1]] : i32, !py.dynamic]
// CHECK-NEXT: return %[[RESULT]]

// -----

py.func @make_object(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeObject %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_object
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[SLOTS:.*]] = type_slots %[[ARG]]
// CHECK-NEXT: %[[LEN:.*]] = tuple_len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[ARG]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initObject %[[MEM]]
// CHECK-NEXT: return %[[RESULT]]

// -----

// CHECK: #[[$TUPLE_TYPE:.*]] = #py.globalValue<builtins.tuple{{,|>}}

py.func @make_tuple_from_list(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = list_toTuple %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_tuple_from_list
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = constant(#[[$TUPLE_TYPE]])
// CHECK-NEXT: %[[SIZE:.*]] = list_len %[[ARG]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[TUPLE]][%[[SIZE]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initTupleFromList %[[MEM]] to (* %[[ARG]])
// CHECK-NEXT: return %[[RESULT]]

// -----

// CHECK: #[[$BOOL_TYPE:.*]] = #py.globalValue<builtins.bool{{,|>}}

py.func @make_bool_from_i1(%arg0 : i1) -> !py.dynamic {
    %0 = bool_fromI1 %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_bool_from_i1
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[EXT:.*]] = arith.extui %[[ARG]] : i1 to i{{[0-9]+}}
// CHECK-NEXT: %[[INDEX:.*]] = arith.index_cast %[[EXT]]
// CHECK-NEXT: %[[BOOL:.*]] = constant(#[[$BOOL_TYPE]])
// CHECK-NEXT: %[[SLOTS:.*]] = type_slots %[[BOOL]]
// CHECK-NEXT: %[[LEN:.*]] = tuple_len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[BOOL]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initIntUnsigned %[[MEM]] to %[[INDEX]]
// CHECK-NEXT: return %[[RESULT]]

// -----

// CHECK: #[[$INT_TYPE:.*]] = #py.globalValue<builtins.int{{,|>}}

py.func @make_int_fromInteger(%arg0 : index) -> !py.dynamic {
    %0 = int_fromUnsigned %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_int_fromInteger
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[BOOL:.*]] = constant(#[[$INT_TYPE]])
// CHECK-NEXT: %[[SLOTS:.*]] = type_slots %[[BOOL]]
// CHECK-NEXT: %[[LEN:.*]] = tuple_len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[BOOL]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initIntUnsigned %[[MEM]] to %[[ARG]]
// CHECK-NEXT: return %[[RESULT]]


// -----

// CHECK: #[[$STR_TYPE:.*]] = #py.globalValue<builtins.str{{,|>}}

py.func @make_str_fromInt(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = int_toStr %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_str_fromInt
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[STR:.*]] = constant(#[[$STR_TYPE]])
// CHECK-NEXT: %[[SLOTS:.*]] = type_slots %[[STR]]
// CHECK-NEXT: %[[LEN:.*]] = tuple_len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[STR]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initStrFromInt %[[MEM]] to %[[ARG]]
// CHECK-NEXT: return %[[RESULT]]

// -----

// CHECK: #[[$INT_TYPE:.*]] = #py.globalValue<builtins.int{{,|>}}

py.func @make_int_from_add(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = int_add %arg0, %arg1
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_int_from_add
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[INT:.*]] = constant(#[[$INT_TYPE]])
// CHECK-NEXT: %[[SLOTS:.*]] = type_slots %[[INT]]
// CHECK-NEXT: %[[LEN:.*]] = tuple_len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[INT]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initIntAdd %[[MEM]] to %[[ARG0]] + %[[ARG1]]
// CHECK-NEXT: return %[[RESULT]]

// -----

// CHECK: #[[$FLOAT_TYPE:.*]] = #py.globalValue<builtins.float{{,|>}}

py.func @make_float_fromF64(%arg0 : f64) -> !py.dynamic {
    %0 = float_fromF64 %arg0
    return %0 : !py.dynamic
}

// CHECK-LABEL: @make_float_fromF64
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[FLOAT:.*]] = constant(#[[$FLOAT_TYPE]])
// CHECK-NEXT: %[[SLOTS:.*]] = type_slots %[[FLOAT]]
// CHECK-NEXT: %[[LEN:.*]] = tuple_len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[FLOAT]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initFloat %[[MEM]] to %[[ARG]]
// CHECK-NEXT: return %[[RESULT]]

// -----

// CHECK-DAG: #[[$TYPE_TYPE:.*]] = #py.globalValue<builtins.type{{,|>}}
// CHECK-DAG: #[[$TUPLE_TYPE:.*]] = #py.globalValue<builtins.tuple{{,|>}}

py.func @make_type(%name : !py.dynamic,
                            %mro : !py.dynamic,
                            %slots : !py.dynamic) -> !py.dynamic {
  %0 = makeType(name=%name, mro=%mro, slots=%slots)
  return %0 : !py.dynamic
}

// CHECK-LABEL: @make_type
// CHECK-SAME: %[[NAME:[[:alnum:]]+]]
// CHECK-SAME: %[[MRO:[[:alnum:]]+]]
// CHECK-SAME: %[[INSTANCE_SLOTS:[[:alnum:]]+]]
// CHECK-NEXT: %[[TYPE:.*]] = constant(#[[$TYPE_TYPE]])
// CHECK-NEXT: %[[SLOTS:.*]] = type_slots %[[TYPE]]
// CHECK-NEXT: %[[LEN:.*]] = tuple_len %[[SLOTS]]
// CHECK-NEXT: %[[MEM:.*]] = pyMem.gcAllocObject %[[TYPE]][%[[LEN]]]
// CHECK-NEXT: %[[TUPLE:.*]] = constant(#[[$TUPLE_TYPE]])
// CHECK-NEXT: %[[MRO_LEN:.*]] = tuple_len %[[MRO]]
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[INC:.*]] = arith.addi %[[MRO_LEN]], %[[ONE]]
// CHECK-NEXT: %[[MRO_MEM:.*]] = pyMem.gcAllocObject %[[TUPLE]][%[[INC]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initType %[[MEM]](name = %[[NAME]], mro = %[[MRO_MEM]] to %[[MRO]], slots = %[[INSTANCE_SLOTS]])
// CHECK-NEXT: return %[[RESULT]]
