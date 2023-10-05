// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func @entry_block(%arg0 : !py.dynamic) -> i1 {
    %0 = isUnboundValue %arg0
    return %0 : i1
}

// CHECK-LABEL: @entry_block
// CHECK: %[[CONST:.*]] = arith.constant false
// CHECK: return %[[CONST]]

// -----

py.global @a : !py.dynamic

py.func @block_argument(%arg0 : i1) -> i1 {
    %c = load @a : !py.dynamic
    cf.cond_br %arg0, ^true, ^false(%c : !py.dynamic)

^true:
    %u = constant(#py.unbound)
    cf.br ^false(%u : !py.dynamic)

^false(%0 : !py.dynamic):
    %1 = isUnboundValue %0
    return %1 : i1
}

// CHECK-LABEL: @block_argument
// ...
// CHECK: %[[I1:.*]] = isUnboundValue
// CHECK: return %[[I1]]

// -----

py.global @a : !py.dynamic

py.func @load_op(%arg0 : !py.dynamic) -> i1 {
    store %arg0 : !py.dynamic into @a
    %0 = load @a : !py.dynamic
    %1 = isUnboundValue %0
    return %1 : i1
}

// CHECK-LABEL: @load_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: store %[[ARG0]] : !py.dynamic into @a
// CHECK: %[[LOADED:.*]] = load @a
// CHECK: %[[UNBOUND:.*]] = isUnboundValue %[[LOADED]]
// CHECK: return %[[UNBOUND]]

// -----

py.func @select_pat1(%r : i1, %arg0 : !py.dynamic, %arg1 : !py.dynamic) -> i1 {
    %0 = arith.select %r, %arg0, %arg1 : !py.dynamic
    %1 = isUnboundValue %0
    return %1 : i1
}

// CHECK-LABEL: @select_pat1
// CHECK-NEXT: %[[RES:.*]] = arith.constant false
// CHECK-NEXT: return %[[RES]]

py.func @select_pat2(%r : i1, %arg0 : !py.dynamic) -> i1 {
    %0 = constant(#py.unbound)
    %1 = arith.select %r, %arg0, %0 : !py.dynamic
    %2 = isUnboundValue %1
    return %2 : i1
}

// CHECK-LABEL: @select_pat2
// CHECK-SAME: %[[R:[[:alnum:]]+]]
// CHECK-NEXT: %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT: %[[INV:.*]] = arith.xori %[[R]], %[[TRUE]]
// CHECK-NEXT: return %[[INV]]

py.func @select_pat3(%r : i1, %arg0 : !py.dynamic) -> i1 {
    %0 = constant(#py.unbound)
    %1 = arith.select %r, %0, %arg0 : !py.dynamic
    %2 = isUnboundValue %1
    return %2 : i1
}

// CHECK-LABEL: @select_pat3
// CHECK-SAME: %[[R:[[:alnum:]]+]]
// CHECK-NEXT: return %[[R]]
