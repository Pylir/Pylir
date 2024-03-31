# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

# CHECK: #[[$IS_INSTANCE:.*]] = #py.globalValue<builtins.isinstance{{>|,}}
# CHECK: #[[$TYPE:.*]] = #py.globalValue<builtins.type{{>|,}}

# CHECK-LABEL: init "__main__"
# CHECK: %[[TYPE_ERROR:.*]] = arith.select
# CHECK: %[[IS_INSTANCE:.*]] = py.constant(#[[$IS_INSTANCE]])
# CHECK: %[[TYPE:.*]] = py.constant(#[[$TYPE]])
# CHECK: %[[CHECK:.*]] = call %[[IS_INSTANCE]](%[[TYPE_ERROR]], %[[TYPE]])
# CHECK: %[[I1:.*]] = py.bool_toI1 %[[CHECK]]
# CHECK: cf.cond_br %[[I1]], ^[[BB1:.*]], ^[[BB2:.*]](%[[TYPE_ERROR]] : !py.dynamic)

# CHECK: ^[[BB1]]:
# CHECK: %[[EXC:.*]] = call %[[TYPE_ERROR]]()
# CHECK: cf.br ^[[BB2]](%[[EXC]] : !py.dynamic)

# CHECK: ^[[BB2]](%[[EXC:.*]]: !py.dynamic loc({{.*}})):
# CHECK: py.raise %[[EXC]]
raise TypeError

try:
    # CHECK: py.raiseEx %{{.*}}
    # CHECK-NEXT: label ^{{.*}} unwind ^{{.*}}
    raise NameError
except NameError:
    pass
