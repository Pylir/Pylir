// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @getitem_string_positive
// CHECK: %[[RES:.*]] = py.constant "x"
// CHECK: return %[[RES]]
func @getitem_string_positive() -> !py.dynamic {
    %0 = py.constant "text"
    %1 = py.constant #py.int<2>
    %2 = py.getItem %0[%1]
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @getitem_string_negative
// CHECK: %[[RES:.*]] = py.constant "t"
// CHECK: return %[[RES]]
func @getitem_string_negative() -> !py.dynamic {
    %0 = py.constant "text"
    %1 = py.constant #py.int<-1>
    %2 = py.getItem %0[%1]
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @getitem_tuple_positive
// CHECK: %[[RES:.*]] = py.constant "x"
// CHECK: return %[[RES]]
func @getitem_tuple_positive() -> !py.dynamic {
    %0 = py.constant #py.tuple<(#py.int<5>,"x",#py.int<3>)>
    %1 = py.constant #py.int<1>
    %2 = py.getItem %0[%1]
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @getitem_tuple_negative
// CHECK: %[[RES:.*]] = py.constant #py.int<3>
// CHECK: return %[[RES]]
func @getitem_tuple_negative() -> !py.dynamic {
    %0 = py.constant #py.tuple<(#py.int<5>,"x",#py.int<3>)>
    %1 = py.constant #py.int<-1>
    %2 = py.getItem %0[%1]
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @getitem_list_positive
// CHECK: %[[RES:.*]] = py.constant "x"
// CHECK: return %[[RES]]
func @getitem_list_positive() -> !py.dynamic {
    %0 = py.constant #py.list<[#py.int<5>,"x",#py.int<3>]>
    %1 = py.constant #py.int<1>
    %2 = py.getItem %0[%1]
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @getitem_list_negative
// CHECK: %[[RES:.*]] = py.constant #py.int<3>
// CHECK: return %[[RES]]
func @getitem_list_negative() -> !py.dynamic {
    %0 = py.constant #py.list<[#py.int<5>,"x",#py.int<3>]>
    %1 = py.constant #py.int<-1>
    %2 = py.getItem %0[%1]
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @getitem_dict
// CHECK: %[[RES:.*]] = py.constant #py.int<3>
// CHECK: return %[[RES]]
func @getitem_dict() -> !py.dynamic {
    %0 = py.constant #py.dict<{#py.int<5> to "x","text" to #py.int<3>}>
    %1 = py.constant "text"
    %2 = py.getItem %0[%1]
    return %2 : !py.dynamic
}
