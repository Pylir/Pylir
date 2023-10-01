# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-DAG: #[[$UNBOUND:.*]] = #py.globalValue<builtins.UnboundLocalError,
# CHECK-DAG: #[[$NAME_ERROR:.*]] = #py.globalValue<builtins.NameError,
def g():
    global a
    del a


# CHECK-LABEL: py.func private @"g$impl[0]"
# CHECK: %[[LOAD:.*]] = load @a$handle
# CHECK: %[[IS_UNBOUND:.*]] = isUnboundValue %[[LOAD]]
# CHECK: cf.cond_br %[[IS_UNBOUND]], ^[[UNBOUND_BLOCK:.*]], ^[[DEL_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[UNBOUND_BLOCK]]:
# CHECK: %[[NAME_ERROR:.*]] = constant(#[[$NAME_ERROR]])
# CHECK: %[[MRO:.*]] = type_mro %[[NAME_ERROR]]
# CHECK: %[[NEW:.*]] = mroLookup %{{.*}} in %[[MRO]]
# CHECK: %[[EXC:.*]] = function_call %[[NEW]](%[[NEW]], %{{.*}}, %{{.*}})
# CHECK: raise %[[EXC]]

# CHECK: ^[[DEL_BLOCK]]:
# CHECK: %[[UNBOUND:.*]] = constant(#py.unbound)
# CHECK: store %[[UNBOUND]] : !py.dynamic into @a$handle

def local():
    del a
    return a


# CHECK-LABEL: py.func private @"local$impl[0]"
# CHECK: %[[A:.*]] = constant(#py.unbound)
# CHECK: %[[IS_UNBOUND:.*]] = isUnboundValue %[[A:.*]]
# CHECK: cf.cond_br %[[IS_UNBOUND]], ^[[UNBOUND_BLOCK:.*]], ^[[DEL_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[UNBOUND_BLOCK]]:
# CHECK: %[[NAME_ERROR:.*]] = constant(#[[$UNBOUND]])
# CHECK: %[[MRO:.*]] = type_mro %[[NAME_ERROR]]
# CHECK: %[[NEW:.*]] = mroLookup %{{.*}} in %[[MRO]]
# CHECK: %[[EXC:.*]] = function_call %[[NEW]](%[[NEW]], %{{.*}}, %{{.*}})
# CHECK: raise %[[EXC]]

# CHECK: ^[[DEL_BLOCK]]:
# CHECK: %[[UNBOUND:.*]] = constant(#py.unbound)
# CHECK: return %[[UNBOUND]]

def closure():
    a = 0

    def inner():
        nonlocal a
        del a

# CHECK-LABEL: py.func private @"closure.<locals>.inner$impl[0]"
# CHECK-SAME: %[[FUNC_OBJ:[[:alnum:]]+]]
# CHECK: %[[CLOSURE:.*]] = getSlot %[[FUNC_OBJ]][%{{.*}}]
# CHECK: %[[ZERO:.*]] = arith.constant 0
# CHECK: %[[A_CELL:.*]] = tuple_getItem %[[CLOSURE]][%[[ZERO]]]
# CHECK: %[[A:.*]] = getSlot %[[A_CELL]][%{{.*}}]
# CHECK: %[[IS_UNBOUND:.*]] = isUnboundValue %[[A:.*]]
# CHECK: cf.cond_br %[[IS_UNBOUND]], ^[[UNBOUND_BLOCK:.*]], ^[[DEL_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[UNBOUND_BLOCK]]:
# CHECK: %[[NAME_ERROR:.*]] = constant(#[[$UNBOUND]])
# CHECK: %[[MRO:.*]] = type_mro %[[NAME_ERROR]]
# CHECK: %[[NEW:.*]] = mroLookup %{{.*}} in %[[MRO]]
# CHECK: %[[EXC:.*]] = function_call %[[NEW]](%[[NEW]], %{{.*}}, %{{.*}})
# CHECK: raise %[[EXC]]

# CHECK: ^[[DEL_BLOCK]]:
# CHECK: %[[UNBOUND:.*]] = constant(#py.unbound)
# CHECK: setSlot %[[A_CELL]][%{{.*}}] to %[[UNBOUND]]
