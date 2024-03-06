# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$HASH:.*]] = #py.globalValue<builtins.hash{{,|>}}

# CHECK-LABEL: init "__main__"

# CHECK: py.makeDict ()
{}


# CHECK-LABEL: func "__main__.foo"
# CHECK-SAME: %[[X:[[:alnum:]]+]]
def foo(x):
    # CHECK: %[[SEVEN:.*]] = py.constant(#py.int<7>)
    # CHECK: %[[EIGHT:.*]] = py.constant(#py.int<8>)
    # CHECK: %[[HASH_REF:.*]] = py.constant(#[[$HASH]])
    # CHECK: %[[HASH:.*]] = call %[[HASH_REF]](%[[SEVEN]])
    # CHECK: %[[INT:.*]] = py.int_toIndex %[[HASH]]
    # CHECK: py.makeDict (%[[SEVEN]] hash(%[[INT]]) : %[[EIGHT]], **%[[X]])
    return {7: 8, **x}

