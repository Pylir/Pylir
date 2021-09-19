# RUN: pylir %s -emit-mlir -o - | FileCheck %s


def foo():
    def bar():
        def foobar():
            pass


# CHECK-DAG: func private @"foo$impl[0]"
# CHECK-DAG: func private @"foo.<locals>.bar$impl[0]"
# CHECK-DAG: func private @"foo.<locals>.bar.<locals>.foobar$impl[0]"

class Foo:
    def bar(self):
        pass


# CHECK-DAG: func private @"Foo.bar$impl[0]"


def a():
    class Foo:
        pass

# CHECK: %[[NAME:.*]] = py.constant "a.<locals>.Foo"
# CHECK: py.makeClass %[[NAME]]
