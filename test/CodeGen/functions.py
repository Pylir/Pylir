# RUN: pylir %s -emit-mlir -o - | FileCheck %s

# CHECK: @foo

# CHECK-LABEL: __init__

# CHECK: %[[RES:.*]] = py.makeFunc @"foo$impl[0]"
# CHECK: %[[NAME:.*]] = py.constant "foo"
# CHECK: py.setAttr "__name__" of %[[RES]] to %[[NAME]]
# CHECK: %[[NAME:.*]] = py.constant "foo"
# CHECK: py.setAttr "__qualname__" of %[[RES]] to %[[NAME]]
# CHECK: %[[DEFAULTS:.*]] = py.singleton None
# CHECK: py.setAttr "__defaults__" of %[[RES]] to %[[DEFAULTS]]
# CHECK: %[[KWDEFAULTS:.*]] = py.singleton None
# CHECK: py.setAttr "__kwdefaults__" of %[[RES]] to %[[KWDEFAULTS]]
# CHECK: %[[FOO:.*]] = py.getGlobal @foo
# CHECK: py.store %[[RES]] into %[[FOO]]

def foo():
    x = 3

    def bar(a=3, *, c=1):
        return a + c

# CHECK-LABEL: func private @"foo$impl[0]"

# CHECK: %[[X:.*]] = py.alloca
# CHECK: %[[BAR:.*]] = py.alloca
# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: py.store %[[THREE]] into %[[X]]

# CHECK: %[[THREE:.*]] = py.constant #py.int<3>
# CHECK: %[[ONE:.*]] = py.constant #py.int<1>
# CHECK: %[[C:.*]] = py.constant "c"
# CHECK: %[[RES:.*]] = py.makeFunc @"foo.<locals>.bar$impl[0]"
# CHECK: %[[NAME:.*]] = py.constant "bar"
# CHECK: py.setAttr "__name__" of %[[RES]] to %[[NAME]]
# CHECK: %[[NAME:.*]] = py.constant "foo.<locals>.bar"
# CHECK: py.setAttr "__qualname__" of %[[RES]] to %[[NAME]]
# CHECK: %[[DEFAULTS:.*]] = py.makeTuple (%[[THREE]])
# CHECK: py.setAttr "__defaults__" of %[[RES]] to %[[DEFAULTS]]
# CHECK: %[[KWDEFAULTS:.*]] = py.makeDict (%[[C]] : %[[ONE]])
# CHECK: py.setAttr "__kwdefaults__" of %[[RES]] to %[[KWDEFAULTS]]
# CHECK: py.store %[[RES]] into %[[BAR]]

# CHECK: func private @"foo.<locals>.bar$impl[0]"
# CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
# CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
# CHECK: %[[a:.*]] = py.alloc
# CHECK: py.store %[[ARG0]] into %[[a]]
# CHECK: %[[c:.*]] = py.alloc
# CHECK: py.store %[[ARG1]] into %[[c]]
