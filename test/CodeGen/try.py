# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$INSTANCE_OF:.*]] = #py.globalValue<builtins.isinstance{{>|,}}

# CHECK-LABEL: func "__main__.a_call"
def a_call():
    try:
        # CHECK: callEx %{{.*}}()
        # CHECK-NEXT: label ^[[NORMAL:[[:alnum:]]+]] unwind ^[[EXCEPT:[[:alnum:]]+]]
        print()
    # CHECK: ^[[NORMAL]]:
    # CHECK: cf.br ^[[ELSE:[[:alnum:]]+]]

    # CHECK: ^[[EXCEPT]](%[[EXC:[[:alnum:]]+]]: !py.dynamic {{.*}}):
    # CHECK: %[[FILTER:.*]] = arith.select
    # CHECK: %[[INSTANCE_OF:.*]] = py.constant(#[[$INSTANCE_OF]])
    # CHECK: %[[BOOL:.*]] = call %[[INSTANCE_OF]](%[[EXC]], %[[FILTER]])
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[BOOL]]
    # CHECK: cf.cond_br %[[I1]], ^[[BODY:.*]], ^[[CONTINUE:[[:alnum:]]+]]
    except StopIteration:
        # CHECK: ^[[BODY:.*]]
        # CHECK: cf.br ^[[THEN:[[:alnum:]]+]]
        pass
    # CHECK: ^[[CONTINUE]]:
    except:
        # CHECK: call
        # CHECK: cf.br ^[[THEN]]
        print()
    # CHECK: ^[[ELSE]]:
    # CHECK: cf.br ^[[THEN]]
    # CHECK: ^[[THEN]]:
    # CHECK: return


# CHECK-LABEL: func "__main__.should_rethrow"
def should_rethrow():
    try:
        # CHECK: callEx %{{.*}}()
        # CHECK-NEXT: unwind ^[[EXCEPT:[[:alnum:]]+]]
        print()
    # CHECK: ^[[EXCEPT]](%[[EXC:[[:alnum:]]+]]: !py.dynamic {{.*}}):
    # CHECK: cf.cond_br %{{.*}}, ^{{.*}}, ^[[CONTINUE:[[:alnum:]]+]]
    except StopIteration:
        pass
    # CHECK: ^[[CONTINUE]]:
    # CHECK: raise %[[EXC]]
