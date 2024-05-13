# RUN: pylir %s -fsyntax-only -dump-ast | FileCheck %s

# CHECK: globals: foo, x, outer, outer2, outer3, bar, Foo

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
# CHECK-NEXT: cells: c, a

# CHECK-LABEL: function inner
# CHECK-NEXT: parameter b
# CHECK-NEXT: locals: b
# CHECK-NEXT: nonlocals: c, a
# CHECK-NOT: cells:

def outer():
    def inner():
        def inner2():
            nonlocal x

    x = 3


# CHECK-LABEL: function outer
# CHECK-NEXT: locals: inner
# CHECK-NEXT: cells: x

# CHECK-LABEL: function inner
# CHECK-NEXT: locals: inner2
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: cells:
# CHECK-LABEL: function inner2
# CHECK-NOT: locals:
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: cells:

x = 0


def outer2():
    x = 3

    def inner():
        print(x)


# CHECK-LABEL: function outer2
# CHECK-NEXT: locals: inner
# CHECK-NOT: nonlocals
# CHECK-NEXT: cells: x

# CHECK-LABEL: function inner
# CHECK-NOT: locals:
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: cells:

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
# CHECK-NEXT: cells: y
# CHECK-NOT: nonlocals:
# CHECK-LABEL: class Foo
# CHECK-NEXT: locals: foo
# CHECK-NEXT: nonlocals: y
# CHECK-NOT: cells:
# CHECK-LABEL: function foo
# CHECK-NEXT: parameter
# CHECK-NEXT: locals: self
# CHECK-NEXT: nonlocals: y
# CHECK-NOT: cells:

def bar():
    x = 3

    class Bar:
        def outer(self):
            def inner():
                def inner2():
                    nonlocal x


# CHECK-LABEL: function bar
# CHECK-NEXT: locals: Bar
# CHECK-NEXT: cells: x
# CHECK-LABEL: class Bar
# CHECK-NEXT: locals: outer
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: cells:
# CHECK-LABEL: function outer
# CHECK-NEXT: parameter
# CHECK-NEXT: locals: self, inner
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: cells:
# CHECK-LABEL: function inner
# CHECK-NEXT: locals: inner2
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: cells:
# CHECK-LABEL: function inner2
# CHECK-NOT: locals:
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: cells:

class Foo:
    def bar(self):
        return Foo


# CHECK-LABEL: class Foo
# CHECK-NEXT: locals: bar
# CHECK-NOT: cells:
# CHECK-NOT: nonlocals:
# CHECK-LABEL: function bar
# CHECK: locals: self
# CHECK-NOT: cells:
# CHECK-NOT: nonlocals:

def foo():
    x = 3

    def inner():
        global x

        def inner2():
            return x


# CHECK-LABEL: function foo
# CHECK-NEXT: locals: x, inner
# CHECK-NOT: cells:
# CHECK-LABEL: function inner
# CHECK-NEXT: locals: inner2
# CHECK-NOT: nonlocals:
# CHECK-NOT: cells:
# CHECK-LABEL: function inner2
# CHECK-NOT: nonlocals:
# CHECK-NOT: cells:
# CHECK-NOT: locals:

def foo():
    x += 3


# CHECK-LABEL: function foo
# CHECK-NEXT: locals: x

def bar():
    x: 3


# CHECK-LABEL: function bar
# CHECK-NEXT: locals: x

def del_too():
    del a


# CHECK-LABEL: function del_too
# CHECK-NEXT: locals: a

def in_sub():
    x[0] = 3


# CHECK-LABEL: function in_sub
# CHECK-NOT: locals: x

def outer2():
    x = 3

    lambda: print(x)


# CHECK-LABEL: function outer2
# CHECK-NOT: nonlocals
# CHECK-NEXT: cells: x

# CHECK-LABEL: lambda expression
# CHECK-NOT: locals:
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: cells:

lambda x: x

# CHECK-LABEL: lambda expression
# CHECK-NEXT: parameter
# CHECK-NEXT: locals: x
# CHECK-NOT: nonlocals
# CHECK-NOT: cells

lambda: (x := 3)

# CHECK-LABEL: lambda expression
# CHECK-NEXT: locals: x
# CHECK-NOT: nonlocals
# CHECK-NOT: cells

lambda: (x := 3) + (lambda: x)()

# CHECK-LABEL: lambda expression
# CHECK-NOT: locals
# CHECK-NOT: nonlocals
# CHECK-NEXT: cells: x
# CHECK-LABEL: lambda expression
# CHECK-NOT: locals
# CHECK-NEXT: nonlocals: x
# CHECK-NOT: cells


def outer3():
    l = lambda: f
    f = 3

# CHECK-LABEL: function outer3
# CHECK-NEXT: locals: l
# CHECK-NOT: nonlocals
# CHECK-NEXT: cells: f

# CHECK-LABEL: lambda expression
# CHECK-NOT: locals:
# CHECK-NEXT: nonlocals: f
# CHECK-NOT: cells:

def outer4():
    y = 0

    class Test:
        y
# CHECK-LABEL: function outer4
# CHECK: locals: Test
# CHECK-NOT: nonlocals
# CHECK: cells: y
# CHECK-LABEL: class Test
# CHECK-NEXT: nonlocals: y
