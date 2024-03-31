# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: func "__main__.foo"
def foo():
    global x
    try:
        # CHECK: cf.br ^[[ELSE:[[:alnum:]]+]]
        pass
    # CHECK: ^[[ELSE]]:
    finally:
        # CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
        # CHECK: %[[STR:.*]] = py.constant(#py.str<"x">)
        # CHECK: dict_setItem
        # CHECK: cf.br ^[[THEN:[[:alnum:]]+]]
        x = 3
    # CHECK: ^[[THEN]]:
    # CHECK: return


# CHECK-LABEL: func "__main__.foo2"
def foo2():
    global x
    try:
        # CHECK-DAG: %[[FIVE:.*]] = py.constant(#py.int<5>)
        # CHECK-DAG: %[[THREE:.*]] = py.constant(#py.int<3>)
        # CHECK-DAG: %[[STR:.*]] = py.constant(#py.str<"x">)
        # CHECK: dict_setItem %{{.*}}[%[[STR]] hash(%{{.*}})] to %[[THREE]]
        # CHECK-NEXT: return %[[FIVE]]
        return 5
    finally:
        x = 3


# CHECK-LABEL: func "__main__.foo3"
def foo3():
    global x
    try:
        try:
            # CHECK: cf.br ^[[ELSE:[[:alnum:]]+]]
            pass
        finally:
            # CHECK: ^[[ELSE]]:
            # CHECK-DAG: %[[FIVE:.*]] = py.constant(#py.int<5>)
            # CHECK-DAG: %[[STR:.*]] = py.constant(#py.str<"x">)
            # CHECK: dict_setItem %{{.*}}[%[[STR]] hash(%{{.*}})] to %[[FIVE]]
            # CHECK: cf.br ^[[THEN:[[:alnum:]]+]]
            x = 5
        # CHECK: ^[[THEN]]:
        # CHECK: cf.br ^[[ELSE:[[:alnum:]]+]]
        # CHECK: ^[[ELSE]]:
    finally:
        # CHECK-DAG: %[[THREE:.*]] = py.constant(#py.int<3>)
        # CHECK-DAG: %[[STR:.*]] = py.constant(#py.str<"x">)
        # CHECK: dict_setItem %{{.*}}[%[[STR]] hash(%{{.*}})] to %[[THREE]]
        # CHECK: cf.br ^[[THEN:[[:alnum:]]+]]
        x = 3
    # CHECK: ^[[THEN]]:
    # CHECK: return


# CHECK-LABEL: func "__main__.foo4"
def foo4():
    global x
    try:
        # CHECK: cf.cond_br %{{.*}}, ^[[BODY:[[:alnum:]]+]], ^[[ELSE:[[:alnum:]]+]]
        while True:
            # CHECK: ^[[BODY]]:
            try:
                break
            finally:
                # CHECK-DAG: %[[FIVE:.*]] = py.constant(#py.int<5>)
                # CHECK-DAG: %[[STR:.*]] = py.constant(#py.str<"x">)
                # CHECK: dict_setItem %{{.*}}[%[[STR]] hash(%{{.*}})] to %[[FIVE]]
                # CHECK: cf.br ^[[THEN:[[:alnum:]]+]]
                x = 5
        # CHECK: ^[[ELSE]]:
        # CHECK: cf.br ^[[THEN]]
        # CHECK: ^[[THEN]]:
        # CHECK: cf.br ^[[FINALLY:[[:alnum:]]+]]
    finally:
        # CHECK: ^[[FINALLY]]:
        # CHECK-DAG: %[[THREE:.*]] = py.constant(#py.int<3>)
        # CHECK-DAG: %[[STR:.*]] = py.constant(#py.str<"x">)
        # CHECK: dict_setItem %{{.*}}[%[[STR]] hash(%{{.*}})] to %[[THREE]]
        # CHECK: cf.br ^[[THEN:[[:alnum:]]+]]
        x = 3
    # CHECK: ^[[THEN]]:
    # CHECK: return


# CHECK-LABEL: func "__main__.foo5"
def foo5():
    global x
    try:
        try:
            # cf.br ^[[FINALLY:[[:alnum:]]+]]
            pass
        finally:
            # ^[[FINALLY]]:
            # CHECK-DAG: %[[THREE:.*]] = py.constant(#py.int<3>)
            # CHECK-DAG: %[[STR:.*]] = py.constant(#py.str<"x">)
            # CHECK: dict_setItem %{{.*}}[%[[STR]] hash(%{{.*}})] to %[[THREE]]
            # CHECK-NEXT: return
            return
    finally:
        x = 3


# CHECK-LABEL: func "__main__.foo6"
def foo6():
    global x
    try:
        try:
            # CHECK-DAG: %[[FIVE:.*]] = py.constant(#py.int<5>)
            # CHECK-DAG: %[[STR:.*]] = py.constant(#py.str<"x">)
            # CHECK: dict_setItem %{{.*}}[%[[STR]] hash(%{{.*}})] to %[[FIVE]]
            # CHECK-DAG: %[[THREE:.*]] = py.constant(#py.int<3>)
            # CHECK-DAG: %[[STR:.*]] = py.constant(#py.str<"x">)
            # CHECK: dict_setItem %{{.*}}[%[[STR]] hash(%{{.*}})] to %[[THREE]]
            # CHECK-NEXT: return
            return
        finally:
            x = 5
    finally:
        x = 3


# CHECK-LABEL: func "__main__.foo7"
def foo7():
    try:
        # CHECK-LABEL: func "__main__.foo7.<locals>.bar"
        # CHECK-SAME: %[[X:[[:alnum:]]+]]
        def bar(x):
            # CHECK: call %[[X]]()
            x()
            # CHECK-NEXT: %[[ONE:.*]] = py.constant(#py.int<1>)
            # CHECK-NEXT: return %[[ONE]]
            return 1
    finally:
        return 0
