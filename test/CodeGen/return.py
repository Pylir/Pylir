# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$NONE:.*]] = #py.globalValue<builtins.None{{(,|>)}}

def foo():
    return 3


# CHECK-LABEL: func "__main__.foo"
# CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK: return %[[THREE]]

def bar():
    return

# CHECK-LABEL: func "__main__.bar"
# CHECK: %[[NONE:.*]] = py.constant(#[[$NONE]])
# CHECK: return %[[NONE]]
