# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-DAG: #[[$NAME_ERROR:.*]] = #py.globalValue<builtins.NameError{{>|,}}
# CHECK-DAG: #[[$UNBOUND_LOCAL_ERROR:.*]] = #py.globalValue<builtins.UnboundLocalError{{>|,}}

# CHECK-LABEL: init "__main__"

# CHECK-LABEL: func "__main__.foo"
# CHECK-SAME: %[[A:[[:alnum:]]+]]
def foo(a):
    # CHECK: %[[UNBOUND:.*]] = py.isUnboundValue %[[A]]
    # CHECK: cf.cond_br %[[UNBOUND]], ^[[BB1:.*]], ^[[BB2:[[:alnum:]]+]]

    # CHECK: ^[[BB1]]:
    # CHECK: %[[TYPE:.*]] = py.constant(#[[$UNBOUND_LOCAL_ERROR]])
    # CHECK: %[[OBJ:.*]] = call %[[TYPE]]()
    # CHECK: raise %[[OBJ]]
    return a


# CHECK: dict_setItem

# CHECK: %[[STR:.*]] = py.constant(#py.str<"foo">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: %[[FOO:.*]] = py.dict_tryGetItem %{{.*}}[%[[STR]] hash(%[[HASH]])]
# CHECK: %[[UNBOUND:.*]] = py.isUnboundValue %[[FOO]]
# CHECK: cf.cond_br %[[UNBOUND]], ^[[BB1:.*]], ^[[BB2:[[:alnum:]]+]]

# CHECK: ^[[BB1]]:
# CHECK: %[[TYPE:.*]] = py.constant(#[[$NAME_ERROR]])
# CHECK: %[[OBJ:.*]] = call %[[TYPE]]()
# CHECK: raise %[[OBJ]]
foo(0)
