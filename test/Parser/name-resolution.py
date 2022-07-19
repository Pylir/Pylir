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
# CHECK-NEXT: locals: inner
# CHECK-NOT: nonlocals:
# CHECK-NEXT: closures: a, c

# CHECK-LABEL: function inner
# CHECK-NEXT: parameter b
# CHECK-NEXT: locals: b
# CHECK-NEXT: nonlocals: a, c
# CHECK-NOT: closures:

def outer():
    def inner():
        def inner2():
            nonlocal x

    x = 3


# CHECK-LABEL: function outer
# CHECK-NEXT: locals: inner
# CHECK-NEXT: closures: x

# CHECK-LABEL: function inner
# CHECK-NEXT: locals: inner2
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: closures:
# CHECK-LABEL: function inner2
# CHECK-NOT: locals:
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: closures:

x = 0


def outer2():
    x = 3

    def inner():
        print(x)


# CHECK-LABEL: function outer2
# CHECK-NEXT: locals: inner
# CHECK-NOT: nonlocals
# CHECK-NEXT: closures: x

# CHECK-LABEL: function inner
# CHECK-NOT: locals:
# CHECK-NEXT: nonlocals: x
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
# CHECK-NEXT: locals: Foo
# CHECK-NEXT: closures: y
# CHECK-NOT: closures:
# CHECK-LABEL: class Foo
# CHECK-NEXT: locals: foo
# CHECK-NEXT: nonlocals: y
# CHECK-NOT: closures:
# CHECK-LABEL: function foo
# CHECK-NEXT: parameter
# CHECK-NEXT: locals: self
# CHECK-NEXT: nonlocals: y
# CHECK-NOT: closures:

def bar():
    x = 3

    class Bar:
        def outer(self):
            def inner():
                def inner2():
                    nonlocal x

# CHECK-LABEL: function bar
# CHECK-NEXT: locals: Bar
# CHECK-NEXT: closures: x
# CHECK-LABEL: class Bar
# CHECK-NEXT: locals: outer
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: closures:
# CHECK-LABEL: function outer
# CHECK-NEXT: parameter
# CHECK-NEXT: locals: inner, self
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: closures:
# CHECK-LABEL: function inner
# CHECK-NEXT: locals: inner2
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: closures:
# CHECK-LABEL: function inner2
# CHECK-NOT: locals:
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: closures:
