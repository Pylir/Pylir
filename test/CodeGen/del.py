# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def g():
    global a
    del a


# CHECK-LABEL: func.func private @"g$impl[0]"
# CHECK: %[[LOAD:.*]] = py.load @a$handle
# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[LOAD]]
# CHECK: cf.cond_br %[[IS_UNBOUND]], ^[[UNBOUND_BLOCK:.*]], ^[[DEL_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[UNBOUND_BLOCK]]:
# CHECK: %[[NAME_ERROR:.*]] = py.constant(@builtins.NameError)
# CHECK: %[[MRO:.*]] = py.type.mro %[[NAME_ERROR]]
# CHECK: %[[NEW:.*]], %{{.*}} = py.mroLookup "__new__" in %[[MRO]]
# CHECK: %[[EXC:.*]] = py.function.call %[[NEW]](%[[NEW]], %{{.*}}, %{{.*}})
# CHECK: py.raise %[[EXC]]

# CHECK: ^[[DEL_BLOCK]]:
# CHECK: %[[UNBOUND:.*]] = py.constant(#py.unbound)
# CHECK: py.store %[[UNBOUND]] into @a$handle

def local():
    del a
    return a


# CHECK-LABEL: func.func private @"local$impl[0]"
# CHECK: %[[A:.*]] = py.constant(#py.unbound)
# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[A:.*]]
# CHECK: cf.cond_br %[[IS_UNBOUND]], ^[[UNBOUND_BLOCK:.*]], ^[[DEL_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[UNBOUND_BLOCK]]:
# CHECK: %[[NAME_ERROR:.*]] = py.constant(@builtins.UnboundLocalError)
# CHECK: %[[MRO:.*]] = py.type.mro %[[NAME_ERROR]]
# CHECK: %[[NEW:.*]], %{{.*}} = py.mroLookup "__new__" in %[[MRO]]
# CHECK: %[[EXC:.*]] = py.function.call %[[NEW]](%[[NEW]], %{{.*}}, %{{.*}})
# CHECK: py.raise %[[EXC]]

# CHECK: ^[[DEL_BLOCK]]:
# CHECK: %[[UNBOUND:.*]] = py.constant(#py.unbound)
# CHECK: return %[[UNBOUND]]

def closure():
    a = 0

    def inner():
        nonlocal a
        del a

# CHECK-LABEL: func.func private @"closure.<locals>.inner$impl[0]"
# CHECK-SAME: %[[FUNC_OBJ:[[:alnum:]]+]]
# CHECK: %[[CLOSURE:.*]] = py.getSlot "__closure__" from %[[FUNC_OBJ]]
# CHECK: %[[ZERO:.*]] = arith.constant 0
# CHECK: %[[A_CELL:.*]] = py.tuple.getItem %[[CLOSURE]][%[[ZERO]]]
# CHECK: %[[A:.*]] = py.getSlot "cell_contents" from %[[A_CELL]]
# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[A:.*]]
# CHECK: cf.cond_br %[[IS_UNBOUND]], ^[[UNBOUND_BLOCK:.*]], ^[[DEL_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[UNBOUND_BLOCK]]:
# CHECK: %[[NAME_ERROR:.*]] = py.constant(@builtins.UnboundLocalError)
# CHECK: %[[MRO:.*]] = py.type.mro %[[NAME_ERROR]]
# CHECK: %[[NEW:.*]], %{{.*}} = py.mroLookup "__new__" in %[[MRO]]
# CHECK: %[[EXC:.*]] = py.function.call %[[NEW]](%[[NEW]], %{{.*}}, %{{.*}})
# CHECK: py.raise %[[EXC]]

# CHECK: ^[[DEL_BLOCK]]:
# CHECK: %[[UNBOUND:.*]] = py.constant(#py.unbound)
# CHECK: py.setSlot "cell_contents" of %[[A_CELL]] : %{{.*}} to %[[UNBOUND]]
