# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

a = 3
b = 4
a, b = b, a
print(a, b)
# CHECK: 4 3

try:
    a, b = (3,)
    print("Failure")
except ValueError:
    print("Success")
# CHECK-NOT: Failure
# CHECK: Success

try:
    a, b = (3, 4, 5)
    print("Failure")
except ValueError:
    print("Success")
# CHECK-NOT: Failure
# CHECK: Success

*a, b, c = (3, 4, 5, 6)
print(type(a) is list)
print(len(a))
print(a[0], a[1], b, c)
# CHECK: True
# CHECK: 2
# CHECK: 3 4 5 6
# TODO: print list once it has __repr__

*a, b = 3,
print(len(a))
print(b)
# CHECK: 0
# CHECK: 3

try:
    a, *b, c = (3,)
    print("Failure")
except ValueError:
    print("Success")
# CHECK-NOT: Failure
# CHECK: Success
