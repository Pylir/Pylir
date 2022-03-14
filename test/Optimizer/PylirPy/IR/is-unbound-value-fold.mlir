// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.bool = #py.type<>

func @entry_block(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.isUnboundValue %arg0
    %1 = py.bool.fromI1 %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: @entry_block
// CHECK: %[[CONST:.*]] = py.constant #py.bool<value = False>
// CHECK: return %[[CONST]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.bool = #py.type<>

func @block_argument(%arg0 : i1) -> !py.dynamic {
    %c = py.constant #py.bool<value = False>
    cf.cond_br %arg0, ^true, ^false(%c : !py.dynamic)

^true:
    %u = py.constant #py.unbound
    cf.br ^false(%u : !py.dynamic)

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

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.bool = #py.type<>

func @normal_op(%arg0 : () -> !py.dynamic) -> !py.dynamic {
    %0 = call_indirect %arg0() : () -> !py.dynamic
    %1 = py.isUnboundValue %0
    %2 = py.bool.fromI1 %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: @normal_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[C:.*]] = py.constant #py.bool<value = False>
// CHECK: call_indirect %arg0
// CHECK: return %[[C]]

// -----

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.bool = #py.type<>

py.globalHandle @a

func @load_op(%arg0 : !py.dynamic) -> !py.dynamic {
    py.store %arg0 into @a
    %0 = py.load @a
    %1 = py.isUnboundValue %0
    %2 = py.bool.fromI1 %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: @load_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: py.store %[[ARG0]] into @a
// CHECK: %[[LOADED:.*]] = py.load @a
// CHECK: %[[UNBOUND:.*]] = py.isUnboundValue %[[LOADED]]
// CHECK: %[[RESULT:.*]] = py.bool.fromI1 %[[UNBOUND]]
// CHECK: return %[[RESULT]]
