# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-DAG: #[[$ITER:.*]] = #py.globalValue<builtins.iter{{>|,}}
# CHECK-DAG: #[[$NEXT:.*]] = #py.globalValue<builtins.next{{>|,}}
# CHECK-DAG: #[[$ISINSTANCE:.*]] = #py.globalValue<builtins.isinstance{{>|,}}
# CHECK-DAG: #[[$STOPITERATION:.*]] = #py.globalValue<builtins.StopIteration{{>|,}}

# CHECK-LABEL: func "__main__.test"
# CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
def test(iter):
    # CHECK: %[[ITER:.*]] = py.constant(#[[$ITER]])
    # CHECK: %[[ITERATOR:.*]] = call %[[ITER]](%[[ARG0]])
    # CHECK: cf.br ^[[CONDITION:[[:alnum:]]+]]
    # CHECK: ^[[CONDITION]]:
    # CHECK: %[[NEXT:.*]] = py.constant(#[[$NEXT]])
    # CHECK: %[[I:.*]] = callEx %[[NEXT]](%[[ITERATOR]])
    # CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[CATCH:[[:alnum:]]+]]
    # CHECK: ^[[CONTINUE]]:
    # CHECK: cf.br ^[[BODY:[[:alnum:]]+]]
    for i in iter:
        # CHECK: ^[[BODY]]:
        # CHECK: call %{{.*}}(%[[I]])
        # CHECK: cf.br ^[[CONDITION]]
        print(i)
    # CHECK: ^[[CATCH]](%[[EXC:.*]]: !py.dynamic {{.*}}):
    # CHECK-DAG: %[[ISINSTANCE:.*]] = py.constant(#[[$ISINSTANCE]])
    # CHECK-DAG: %[[STOPITERATION:.*]] = py.constant(#[[$STOPITERATION]])
    # CHECK: %[[BOOL:.*]] = call %[[ISINSTANCE]](%[[EXC]], %[[STOPITERATION]])
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
    # CHECK: cf.cond_br %[[I1]], ^[[ELSE:.*]], ^[[RERAISE:[[:alnum:]]+]]
    # CHECK: ^[[RERAISE]]:
    # CHECK: raise %[[EXC]]
    else:
        # CHECK: ^[[ELSE]]:
        # CHECK: call %{{.*}}()
        # CHECK: cf.br ^[[THEN:[[:alnum:]]+]]
        print()
    # CHECK: ^[[THEN]]:
    # CHECK: return


# CHECK-LABEL: func "__main__.break_for"
def break_for(iter):
    # CHECK: %[[I:.*]] = callEx %{{[[:alnum:]]+}}(%{{[[:alnum:]]+}})
    # CHECK-NEXT: label ^[[CONTINUE:.*]] unwind ^[[CATCH:[[:alnum:]]+]]
    # CHECK: ^[[CONTINUE]]:
    # CHECK: cf.br ^[[BODY:[[:alnum:]]+]]
    for i in iter:
        # CHECK: ^[[BODY]]:
        # CHECK: cf.br ^[[THEN:[[:alnum:]]+]]
        break
    # CHECK: ^[[CATCH]](%[[EXC:.*]]: !py.dynamic {{.*}}):
    # CHECK: cf.cond_br %{{[[:alnum:]]+}}, ^[[ELSE:.*]], ^{{[[:alnum:]]+}}
    # CHECK: ^[[ELSE]]:
    # CHECK: cf.br ^[[THEN]]
    # CHECK: ^[[THEN]]:
    # CHECK: return


# CHECK-LABEL: func "__main__.continue_for"
def continue_for(iter):
    # CHECK: cf.br ^[[CONDITION:[[:alnum:]]+]]
    # CHECK: ^[[CONDITION]]:
    # CHECK: %[[I:.*]] = callEx %{{[[:alnum:]]+}}(%{{[[:alnum:]]+}})
    # CHECK-NEXT: label ^[[CONTINUE:[[:alnum:]]+]]
    # CHECK: ^[[CONTINUE]]:
    # CHECK: cf.br ^[[BODY:[[:alnum:]]+]]
    for i in iter:
        # CHECK: ^[[BODY]]:
        # CHECK: cf.br ^[[CONDITION]]
        continue
