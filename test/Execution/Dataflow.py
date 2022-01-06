# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

def foo(x):
    y = 3
    if x:
        y = 5
    return y


# CHECK: 5
# CHECK: 3
print(foo(True))
print(foo(False))


def do_raise():
    raise StopIteration


def foo(x):
    y = 7
    try:
        if x:
            do_raise()
    except StopIteration:
        y = 13
    return y


# CHECK: 13
# CHECK: 7
print(foo(True))
print(foo(False))


def foo(x):
    y = 11
    while x:
        y = 23
        x = False
    return y


# CHECK: 23
# CHECK: 11
print(foo(True))
print(foo(False))
