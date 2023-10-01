# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$OBJECT:.*]] = #py.globalValue<builtins.object,

# CHECK-LABEL: func @__init__

object

# CHECK: constant(#[[$OBJECT]])

# CHECK: %[[VALUE:.*]] = load @BaseException
# CHECK: store %[[VALUE]] : !py.dynamic into @x
x = BaseException

BaseException = True


def foo():
    TypeError = 5
    return TypeError

# CHECK-LABEL: func private @"foo$impl[0]"

# CHECK: %[[FIVE:.*]] = constant(#py.int<5>)
# CHECK: return %[[FIVE]]
