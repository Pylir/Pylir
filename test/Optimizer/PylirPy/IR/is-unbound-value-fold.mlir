// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.bool = #py.type

func.func @entry_block(%arg0 : !py.dynamic) -> i1 {
    %0 = py.isUnboundValue %arg0
    return %0 : i1
}

// CHECK-LABEL: @entry_block
// CHECK: %[[CONST:.*]] = arith.constant false
// CHECK: return %[[CONST]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.bool = #py.type

py.global @a : !py.dynamic

func.func @block_argument(%arg0 : i1) -> i1 {
    %c = py.load @a : !py.dynamic
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
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.bool = #py.type

py.global @a : !py.dynamic

func.func @load_op(%arg0 : !py.dynamic) -> i1 {
    py.store %arg0 : !py.dynamic into @a
    %0 = py.load @a : !py.dynamic
    %1 = py.isUnboundValue %0
    return %1 : i1
}

// CHECK-LABEL: @load_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: py.store %[[ARG0]] : !py.dynamic into @a
// CHECK: %[[LOADED:.*]] = py.load @a
// CHECK: %[[UNBOUND:.*]] = py.isUnboundValue %[[LOADED]]
// CHECK: return %[[UNBOUND]]

// -----

func.func @select_pat1(%r : i1, %arg0 : !py.dynamic, %arg1 : !py.dynamic) -> i1 {
    %0 = arith.select %r, %arg0, %arg1 : !py.dynamic
    %1 = py.isUnboundValue %0
    return %1 : i1
}

// CHECK-LABEL: @select_pat1
// CHECK-NEXT: %[[RES:.*]] = arith.constant false
// CHECK-NEXT: return %[[RES]]

func.func @select_pat2(%r : i1, %arg0 : !py.dynamic) -> i1 {
    %0 = py.constant(#py.unbound)
    %1 = arith.select %r, %arg0, %0 : !py.dynamic
    %2 = py.isUnboundValue %1
    return %2 : i1
}

// CHECK-LABEL: @select_pat2
// CHECK-SAME: %[[R:[[:alnum:]]+]]
// CHECK-NEXT: %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT: %[[INV:.*]] = arith.xori %[[R]], %[[TRUE]]
// CHECK-NEXT: return %[[INV]]

func.func @select_pat3(%r : i1, %arg0 : !py.dynamic) -> i1 {
    %0 = py.constant(#py.unbound)
    %1 = arith.select %r, %0, %arg0 : !py.dynamic
    %2 = py.isUnboundValue %1
    return %2 : i1
}

// CHECK-LABEL: @select_pat3
// CHECK-SAME: %[[R:[[:alnum:]]+]]
// CHECK-NEXT: return %[[R]]
