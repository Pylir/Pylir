# RUN: pylir %s -fsyntax-only -emit-ast 2>&1 | FileCheck %s

def foo():
    global x
    c = 3

    def inner(b):
        nonlocal c
        c = b + a

    a = 0


# CHECK: globals: x

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
