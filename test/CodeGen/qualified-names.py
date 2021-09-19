# RUN: pylir %s -emit-mlir -o - | FileCheck %s

def a():
    class Foo:
        pass


# CHECK: %[[NAME:.*]] = py.constant "a.<locals>.Foo"
# CHECK: py.makeClass %[[NAME]]

def foo():
    def bar():
        def foobar():
            pass


# CHECK-DAG: func private foo
# CHECK-DAG: func private foo.<locals>.bar
# CHECK-DAG: func private foo.<locals>.bar.<locals>.foobar

class Foo:
    def bar(self):
        pass

# CHECK-DAG: func private Foo.bar
