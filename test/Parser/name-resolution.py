# RUN: pylir %s -fsyntax-only -emit-ast 2>&1 | FileCheck %s

# CHECK: globals: bar, foo, outer, outer2, outer3, x

def foo():
    global x
    c = 3

    def inner(b):
        nonlocal c
        c = b + a

    a = 0


# CHECK-LABEL: function foo
# CHECK: locals: a, c
# CHECK-NOT: nonlocals:

# CHECK-LABEL: function inner
# CHECK: locals: b
# CHECK: nonlocals: a, c

def outer():
    def inner():
        def inner2():
            nonlocal x

    x = 3


# CHECK-LABEL: function outer
# CHECK: locals: inner, x

# CHECK-LABEL: function inner
# CHECK: locals: inner2
# CHECK: nonlocals: x
# CHECk_LABEL: function inner2
# CHECK-NOT: locals:
# CHECK: nonlocals: x

x = 0


def outer2():
    x = 3

    def inner():
        print(x)


# CHECK-LABEL: function outer2
# CHECK: locals: inner, x

# CHECK-LABEL: function inner
# CHECK-NOT: locals:
# CHECK: nonlocals: x

def outer3():
    y = 3

    class Foo:
        def foo(self):
            global x
            x = 5
            nonlocal y
            y = self


# CHECK-LABEL: function outer3
# CHECK: locals: Foo, y
# CHECK-LABEL: class Foo
# CHECK: locals: foo
# CHECK: nonlocals: y
# CHECK-LABEL: function foo
# CHECK: locals: self
# CHECK: nonlocals: y

def bar():
    x = 3

    class Bar:
        def outer(self):
            def inner():
                def inner2():
                    nonlocal x

# CHECK-LABEL: function bar
# CHECK: locals: Bar, x
# CHECK-LABEL: class Bar
# CHECK: locals: outer
# CHECK: nonlocals: x
# CHECK-LABEL: function outer
# CHECK: locals: inner, self
# CHECK-LABEL: function inner
# CHECK: locals: inner2
# CHECK: nonlocals: x
# CHECk_LABEL: function inner2
# CHECK-NOT: locals:
# CHECK: nonlocals: x
