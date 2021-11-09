// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

func @entry_block(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.isUnboundValue %arg0
    %1 = py.bool.fromI1 %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @entry_block
// CHECK: %[[CONST:.*]] = py.constant #py.bool<False>
// CHECK: return %[[CONST]]

// -----

func @block_argument(%arg0 : i1) -> !py.dynamic {
    %c = py.constant #py.bool<False>
    cond_br %arg0, ^true, ^false(%c : !py.dynamic)

^true:
    %u = py.constant #py.unbound
    br ^false(%u : !py.dynamic)

^false(%0 : !py.dynamic):
    %1 = py.isUnboundValue %0
    %2 = py.bool.fromI1 %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: @block_argument
// ...
// CHECK: %[[I1:.*]] = py.isUnboundValue
// CHECK: %[[BOOL:.*]] = py.bool.fromI1 %[[I1]]
// CHECK: return %[[BOOL]]

// -----

func @normal_op(%arg0 : () -> !py.dynamic) -> !py.dynamic {
    %0 = call_indirect %arg0() : () -> !py.dynamic
    %1 = py.isUnboundValue %0
    %2 = py.bool.fromI1 %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: @normal_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[C:.*]] = py.constant #py.bool<False>
// CHECK: call_indirect %arg0
// CHECK: return %[[C]]
