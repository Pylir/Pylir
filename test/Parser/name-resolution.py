# RUN: pylir %s -fsyntax-only -dump-ast | FileCheck %s

# CHECK: globals: bar, foo, outer, outer2, outer3, x

def foo():
    global x
    c = 3

    def inner(b):
        nonlocal c
        c = b + a

    a = 0


# CHECK-LABEL: function foo
# CHECK: locals: inner
# CHECK-NOT: nonlocals:
# CHECK: closures: a, c

# CHECK-LABEL: function inner
# CHECK: locals: b
# CHECK: nonlocals: a, c
# CHECK-NOT: closures:

def outer():
    def inner():
        def inner2():
            nonlocal x

    x = 3


# CHECK-LABEL: function outer
# CHECK: locals: inner
# CHECK: closures: x

# CHECK-LABEL: function inner
# CHECK: locals: inner2
# CHECK: nonlocals: x
# CHECK-NOT: closures:
# CHECk_LABEL: function inner2
# CHECK-NOT: locals:
# CHECK: nonlocals: x
# CHECK-NOT: closures:

x = 0


def outer2():
    x = 3

    def inner():
        print(x)


# CHECK-LABEL: function outer2
# CHECK: locals: inner
# CHECK-NOT: nonlocals
# CHECK: closures: x

# CHECK-LABEL: function inner
# CHECK-NOT: locals:
# CHECK: nonlocals: x
# CHECK-NOT: closures:

def outer3():
    y = 3

    class Foo:
        def foo(self):
            global x
            x = 5
            nonlocal y
            y = self


# CHECK-LABEL: function outer3
# CHECK: locals: Foo
# CHECK: closures: y
# CHECK-NOT: closures:
# CHECK-LABEL: class Foo
# CHECK: locals: foo
# CHECK: nonlocals: y
# CHECK-NOT: closures:
# CHECK-LABEL: function foo
# CHECK: locals: self
# CHECK: nonlocals: y
# CHECK-NOT: closures:

def bar():
    x = 3

    class Bar:
        def outer(self):
            def inner():
                def inner2():
                    nonlocal x

# CHECK-LABEL: function bar
# CHECK: locals: Bar
# CHECK: closures: x
# CHECK-LABEL: class Bar
# CHECK: locals: outer
# CHECK: nonlocals: x
# CHECK-NOT: closures:
# CHECK-LABEL: function outer
# CHECK: locals: inner, self
# CHECK: nonlocals: x
# CHECK-NOT: closures:
# CHECK-LABEL: function inner
# CHECK: locals: inner2
# CHECK: nonlocals: x
# CHECK-NOT: closures:
# CHECk_LABEL: function inner2
# CHECK-NOT: locals:
# CHECK: nonlocals: x
# CHECK-NOT: closures:
