# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines


def outer(x):
    return lambda y: x + y


f = outer(3)
print(f(4))
# CHECK: 7

print(f(10))
# CHECK: 13
