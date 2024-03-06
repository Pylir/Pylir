# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$BOOL:.*]] = #py.globalValue<builtins.bool{{,|>}}

# CHECK-LABEL: init "__main__"

# CHECK: %[[ZERO:.*]] = py.constant(#py.int<0>)
# CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
# CHECK: %[[B:.*]] = call %[[BOOL]](%[[ZERO]])
# CHECK: cf.br ^[[IS_BOOL_BB:.*]](%[[B]] : !py.dynamic)
# CHECK: ^[[IS_BOOL_BB]](%[[BOOL:.*]]: !py.dynamic loc({{.*}})):
# CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
# CHECK: cf.cond_br %[[I1]], ^[[BB1:.*]], ^[[BB2:[[:alnum:]]+]]

# CHECK: ^[[BB1]]:
# CHECK: %[[FIVE:.*]] = py.constant(#py.int<5>)
# CHECK: cf.br ^[[BB3:.*]](%[[FIVE]] : !py.dynamic)

# CHECK: ^[[BB2]]:
# CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK: cf.br ^[[BB3]](%[[THREE]] : !py.dynamic)

# CHECK: ^[[BB3]](%[[VALUE:[[:alnum:]]+]]: !py.dynamic loc({{.*}}):
# CHECK: py.dict_setItem %{{.*}}[%{{.*}}] to %[[VALUE]]
x = 5 if 0 else 3
