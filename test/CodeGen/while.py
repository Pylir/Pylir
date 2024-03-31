# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$BOOL:.*]] = #py.globalValue<builtins.bool{{,|>}}

# CHECK-LABEL: init "__main__"
# CHECK: cf.br ^[[COND:[[:alnum:]]+]]

# CHECK: ^[[COND]]:
# CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
# CHECK: %[[B:.*]] = call %[[BOOL]](%{{.*}})
# CHECK: cf.br ^[[IS_BOOL_BB:.*]](%[[B]] : !py.dynamic)
# CHECK: ^[[IS_BOOL_BB]](%[[BOOL:.*]]: !py.dynamic loc({{.*}})):
# CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
# CHECK: cf.cond_br %[[I1]], ^[[BODY:.*]], ^[[ELSE:[[:alnum:]]+]]
while True:
    # CHECK: ^[[BODY]]:
    # CHECK: call %{{.*}}()
    # CHECK: cf.br ^[[COND]]
    print()
else:
    # CHECK: ^[[ELSE]]:
    # CHECK: call %{{.*}}()
    # CHECK: cf.br ^[[THEN:[[:alnum:]]+]]
    print()


# CHECK: ^[[THEN]]:

# CHECK-LABEL: func "__main__.foo"
# CHECK-SAME: %[[A:[[:alnum:]]+]]
def foo(a):
    # CHECK: cf.br ^[[COND:[[:alnum:]]+]](%[[A]] : !py.dynamic)
    # CHECK: ^[[COND]](%[[A:[[:alnum:]]+]]: !py.dynamic {{.*}}):
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[B:.*]] = call %[[BOOL]](%{{.*}})
    # CHECK: cf.br ^[[IS_BOOL_BB:.*]](%[[B]] : !py.dynamic)
    # CHECK: ^[[IS_BOOL_BB]](%[[BOOL:.*]]: !py.dynamic loc({{.*}})):
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
    # CHECK: cf.cond_br %[[I1]], ^[[BODY:.*]], ^[[ELSE:[[:alnum:]]+]]
    while False:
        # CHECK: ^[[BODY]]:
        # CHECK: call %{{.*}}()
        print(a)
        # CHECK: %[[NEW_A:.*]] = binOp %[[A]] __add__ %{{.*}}
        a = a + 1
        # CHECK: cf.br ^[[COND]](%[[NEW_A]] : !py.dynamic)

    # CHECK: ^[[ELSE]]:
    # CHECK: cf.br ^[[THEN:[[:alnum:]]+]]
    # CHECK: ^[[THEN]]:


# CHECK: cf.br ^[[COND:[[:alnum:]]+]]

# CHECK: ^[[COND]]:
# CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
# CHECK: %[[B:.*]] = call %[[BOOL]](%{{.*}})
# CHECK: cf.br ^[[IS_BOOL_BB:.*]](%[[B]] : !py.dynamic)
# CHECK: ^[[IS_BOOL_BB]](%[[BOOL:.*]]: !py.dynamic loc({{.*}})):
# CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
# CHECK: cf.cond_br %[[I1]], ^[[BODY:.*]], ^[[ELSE:[[:alnum:]]+]]
while True:
    # CHECK: ^[[BODY]]:
    # CHECK: cf.br ^[[THEN:[[:alnum:]]+]]
    break
# CHECK: ^[[ELSE]]:
# CHECK: cf.br ^[[THEN]]

# CHECK: ^[[THEN]]:
# CHECK: cf.br ^[[COND:[[:alnum:]]+]]

# CHECK: ^[[COND]]:
# CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
# CHECK: %[[B:.*]] = call %[[BOOL]](%{{.*}})
# CHECK: cf.br ^[[IS_BOOL_BB:.*]](%[[B]] : !py.dynamic)
# CHECK: ^[[IS_BOOL_BB]](%[[BOOL:.*]]: !py.dynamic loc({{.*}})):
# CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
# CHECK: cf.cond_br %[[I1]], ^[[BODY:.*]], ^[[ELSE:[[:alnum:]]+]]
while True:
    # CHECK: ^[[BODY]]:
    # CHECK: cf.br ^[[COND]]
    continue
