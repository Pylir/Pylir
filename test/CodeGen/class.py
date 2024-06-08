# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

x = 3


def foo():
    # CHECK: %[[$Y_CELL:.*]] = call %{{.*}}()
    y = 5

    # CHECK-LABEL: class "__main__.foo.<locals>.Point" {
    # CHECK-NEXT: %[[DICT:[[:alnum:]]+]]:
    class Point:
        # CHECK: %[[STR:.*]] = py.constant(#py.str<"print">)
        # CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
        # CHECK: %[[LOOKUP:.*]] = py.dict_tryGetItem %[[DICT]][%[[STR]] hash(%[[HASH]])]
        # CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[LOOKUP]]
        # CHECK: cf.cond_br %[[IS_UNBOUND]], ^[[GLOBAL_LOOKUP:.*]], ^[[CONTINUE:.*]](%[[LOOKUP]] : !py.dynamic)

        # CHECK: ^[[GLOBAL_LOOKUP]]:
        # CHECK: %[[STR:.*]] = py.constant(#py.str<"print">)
        # CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
        # CHECK: %[[GLOBAL:.*]] = py.dict_tryGetItem %{{.*}}[%[[STR]] hash(%[[HASH]])]
        # CHECK: %[[BUILTIN:.*]] = py.constant
        # CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[GLOBAL]]
        # CHECK: %[[LOOKUP:.*]] = arith.select %[[IS_UNBOUND]], %[[BUILTIN]], %[[GLOBAL]]
        # CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[LOOKUP]]
        # CHECK: cf.cond_br %[[IS_UNBOUND]], ^[[RAISE:.*]], ^[[CONTINUE2:[[:alnum:]]+]]

        # CHECK: ^[[RAISE]]:
        # CHECK: %[[NAME_ERROR:.*]] = py.constant
        # CHECK: %[[EXC:.*]] = call %[[NAME_ERROR]]()
        # CHECK: py.raise %[[EXC]]

        # CHECK: ^[[CONTINUE2]]:
        # CHECK: cf.br ^[[CONTINUE]](%[[LOOKUP]] : !py.dynamic)

        # CHECK: ^[[CONTINUE]](
        # CHECK-SAME: %[[PRINT:[[:alnum:]]+]]
        # CHECK: %[[STR:.*]] = py.constant(#py.str<"y">)
        # CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
        # CHECK: %[[LOOKUP:.*]] = py.dict_tryGetItem %[[DICT]][%[[STR]] hash(%[[HASH]])]
        # CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[LOOKUP]]
        # CHECK: cf.cond_br %[[IS_UNBOUND]], ^[[NON_LOCAL_LOOKUP:.*]], ^[[CONTINUE:.*]](%[[LOOKUP]] : !py.dynamic)

        # CHECK: ^[[NON_LOCAL_LOOKUP]]:
        # CHECK: %[[ZERO:.*]] = arith.constant 0
        # CHECK: %[[LOOKUP:.*]] = py.getSlot %[[$Y_CELL]][%[[ZERO]]]
        # CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[LOOKUP]]
        # CHECK: cf.cond_br %[[IS_UNBOUND]], ^[[RAISE:.*]], ^[[CONTINUE2:[[:alnum:]]+]]

        # CHECK: ^[[RAISE]]:
        # CHECK: %[[NAME_ERROR:.*]] = py.constant
        # CHECK: %[[EXC:.*]] = call %[[NAME_ERROR]]()
        # CHECK: py.raise %[[EXC]]

        # CHECK: ^[[CONTINUE2]]:
        # CHECK: cf.br ^[[CONTINUE]](%[[LOOKUP]] : !py.dynamic)

        # CHECK: ^[[CONTINUE]](
        # CHECK-SAME: %[[Y:[[:alnum:]]+]]
        # CHECK: call %[[PRINT]](%[[Y]])
        print(y)

        # CHECK: %[[INIT:.*]] = func "__main__.foo.<locals>.Point.__init__"(
        def __init__(self):
            pass
        # CHECK: %[[STR:.*]] = py.constant(#py.str<"__init__">)
        # CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
        # CHECK: py.dict_setItem %[[DICT]][%[[STR]] hash(%[[HASH]])] to %[[INIT]]
        # CHECK: class_return

    try:
        # CHECK-LABEL: classEx "__main__.foo.<locals>.Test" {
        # CHECK: class_return
        # CHECK: } label ^{{.*}} unwind ^{{.*}}
        class Test:
            pass
    except:
        pass
