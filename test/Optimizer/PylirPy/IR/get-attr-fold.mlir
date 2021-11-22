// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func @not_object() -> !py.dynamic {
    %0 = py.constant #py.int<5>
    %result, %success = py.getAttr "size" from %0
    %1 = py.bool.fromI1 %success
    return %1 : !py.dynamic
}

// CHECK-LABEL: @not_object
// CHECK: %[[CONST:.*]] = py.constant #py.bool<False>
// CHECK: return %[[CONST]]

// -----

func @not_contained() -> !py.dynamic {
    %0 = py.constant #py.obj<type: @type, __dict__: #py.dict<{"test" to 3}>>
    %result, %success = py.getAttr "size" from %0
    %1 = py.bool.fromI1 %success
    return %1 : !py.dynamic
}

// CHECK-LABEL: @not_contained
// CHECK: %[[CONST:.*]] = py.constant #py.bool<False>
// CHECK: return %[[CONST]]

// -----

func @contained() -> (!py.dynamic, !py.dynamic) {
    %0 = py.constant #py.obj<type: @type, __dict__: #py.dict<{"size" to #py.int<3>}>>
    %result, %success = py.getAttr "size" from %0
    %1 = py.bool.fromI1 %success
    return %result, %1 : !py.dynamic, !py.dynamic
}

// CHECK-LABEL: @contained
// CHECK-DAG: %[[CONST1:.*]] = py.constant #py.int<3>
// CHECK-DAG: %[[CONST2:.*]] = py.constant #py.bool<True>
// CHECK: return %[[CONST1]], %[[CONST2]]

// -----

py.globalValue @a = #py.int<0>

func @global_value() -> !py.dynamic {
    %0 = py.constant @a
    %result, %success = py.getAttr "__new__" from %0
    %1 = py.bool.fromI1 %success
    return %1 : !py.dynamic
}

// CHECK-LABEL: @global_value
// CHECK: %[[CONSTANT:.*]] = py.constant @a
// CHECK: %[[RESULT:.*]], %[[SUCCESS:.*]] = py.getAttr "__new__" from %[[CONSTANT]]
// CHECK: %[[BOOL:.*]] = py.bool.fromI1 %[[SUCCESS]]
// CHECK: return %[[BOOL]]
