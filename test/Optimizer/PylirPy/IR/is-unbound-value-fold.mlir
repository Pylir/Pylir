// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type

func @entry_block(%arg0 : !py.dynamic) -> i1 {
    %0 = py.isUnboundValue %arg0
    return %0 : i1
}

// CHECK-LABEL: @entry_block
// CHECK: %[[CONST:.*]] = arith.constant false
// CHECK: return %[[CONST]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type

func @block_argument(%arg0 : i1) -> i1 {
    %c = py.constant(#py.bool<False>)
    cf.cond_br %arg0, ^true, ^false(%c : !py.dynamic)

^true:
    %u = py.constant(#py.unbound)
    cf.br ^false(%u : !py.dynamic)

^false(%0 : !py.dynamic):
    %1 = py.isUnboundValue %0
    return %1 : i1
}

// CHECK-LABEL: @block_argument
// ...
// CHECK: %[[I1:.*]] = py.isUnboundValue
// CHECK: return %[[I1]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type

py.globalHandle @a

func @load_op(%arg0 : !py.dynamic) -> i1 {
    py.store %arg0 into @a
    %0 = py.load @a
    %1 = py.isUnboundValue %0
    return %1 : i1
}

// CHECK-LABEL: @load_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: py.store %[[ARG0]] into @a
// CHECK: %[[LOADED:.*]] = py.load @a
// CHECK: %[[UNBOUND:.*]] = py.isUnboundValue %[[LOADED]]
// CHECK: return %[[UNBOUND]]
