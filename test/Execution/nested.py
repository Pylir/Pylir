# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines


def outer(x):
    def inner():
        nonlocal x
        x += 1
        return x

    return inner


f = outer(3)
print(f())
# CHECK: 4

print(f())
# CHECK: 5
