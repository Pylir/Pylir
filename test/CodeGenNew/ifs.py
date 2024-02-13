# RUN: pylir %s -Xnew-codegen -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$BOOL:.*]] = #py.globalValue<builtins.bool{{,|>}}

# CHECK-LABEL: init "__main__"

# CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
# CHECK: %[[B:.*]] = call %[[BOOL]](%{{.*}})
# CHECK: %[[I1:.*]] = py.bool_toI1 %[[B]]
# CHECK: cf.cond_br %[[I1]], ^[[BB1:.*]], ^[[BB2:[[:alnum:]]+]]
if 0:
    # CHECK: ^[[BB1]]:
    # CHECK: call %{{.*}}()
    print()
    # CHECK: cf.br ^[[BB3:[[:alnum:]]+]]

# CHECK: ^[[BB2]]:
# CHECK-NEXT: cf.br ^[[BB3]]

# CHECK: ^[[BB3]]:
# CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
# CHECK: %[[B:.*]] = call %[[BOOL]](%{{.*}})
# CHECK: %[[I1:.*]] = py.bool_toI1 %[[B]]
# CHECK: cf.cond_br %[[I1]], ^[[BB4:.*]], ^[[BB5:[[:alnum:]]+]]
if 0:
    # CHECK: ^[[BB4]]:
    # CHECK: call %{{.*}}()
    # CHECK: cf.br ^[[BB6:[[:alnum:]]+]]
    print()
else:
    # CHECK: ^[[BB5]]:
    # CHECK: call %{{.*}}()
    # CHECK: cf.br ^[[BB6]]
    print()

# CHECK: ^[[BB6]]:
# CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
# CHECK: %[[B:.*]] = call %[[BOOL]](%{{.*}})
# CHECK: %[[I1:.*]] = py.bool_toI1 %[[B]]
# CHECK: cf.cond_br %[[I1]], ^[[BB7:.*]], ^[[BB8:[[:alnum:]]+]]
if 0:
    # CHECK: ^[[BB7]]:
    # CHECK: call %{{.*}}()
    # CHECK: cf.br ^[[BB11:[[:alnum:]]+]]
    print()

    # CHECK: ^[[BB8]]:
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[B:.*]] = call %[[BOOL]](%{{.*}})
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[B]]
    # CHECK: cf.cond_br %[[I1]], ^[[BB9:.*]], ^[[BB10:[[:alnum:]]+]]
elif 1:
    # CHECK: ^[[BB9]]:
    # CHECK: call %{{.*}}()
    # CHECK: cf.br ^[[BB11]]
    print()
else:
    # CHECK: ^[[BB10]]:
    # CHECK: call %{{.*}}()
    # CHECK: cf.br ^[[BB11]]
    print()

# CHECK: ^[[BB11]]:
# CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
# CHECK: %[[B:.*]] = call %[[BOOL]](%{{.*}})
# CHECK: %[[I1:.*]] = py.bool_toI1 %[[B]]
# CHECK: cf.cond_br %[[I1]], ^[[BB12:.*]], ^[[BB13:[[:alnum:]]+]]
if 0:
    # CHECK: ^[[BB12]]:
    # CHECK: call %{{.*}}()
    # CHECK: cf.br ^[[BB16:[[:alnum:]]+]]
    print()

    # CHECK: ^[[BB13]]:
    # CHECK: %[[BOOL:.*]] = py.constant(#[[$BOOL]])
    # CHECK: %[[B:.*]] = call %[[BOOL]](%{{.*}})
    # CHECK: %[[I1:.*]] = py.bool_toI1 %[[B]]
    # CHECK: cf.cond_br %[[I1]], ^[[BB14:.*]], ^[[BB15:[[:alnum:]]+]]
elif 1:
    # CHECK: ^[[BB14]]:
    # CHECK: call %{{.*}}()
    # CHECK: cf.br ^[[BB16]]
    print()

# CHECK: ^[[BB15]]:
# CHECK-NEXT: cf.br ^[[BB16]]

# CHECK: ^[[BB16]]:
# CHECK: init_return
