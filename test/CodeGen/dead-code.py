# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$NONE:.*]] = #py.globalValue<builtins.None{{(,|>)}}

# CHECK-LABEL: func "__main__.foo"
def foo():
    # CHECK: %[[CONSTANT:.*]] = py.constant(#[[$NONE]])
    # CHECK: return %[[CONSTANT]]

    # CHECK-NEXT: ^{{[[:alnum:]]+}}:
    # CHECK: %[[ONE:.*]] = py.constant(#py.int<1>)
    # CHECK: %[[TWO:.*]] = py.constant(#py.int<2>)
    # CHECK: %[[OP:.*]] = binOp %[[ONE]] __add__ %[[TWO]]
    # CHECK: %[[CONSTANT:.*]] = py.constant(#[[$NONE]])
    # CHECK: return %[[CONSTANT]]
    return
    1 + 2
