# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s


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

# CHECK: %[[NAME:.*]] = py.constant(#py.str<"a.<locals>.Foo">)
# TODO:
# COM: CHECK: py.makeClass %[[NAME]]
