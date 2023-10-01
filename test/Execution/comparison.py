# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

print(object() == object())
print(object() != object())
# CHECK: False
# CHECK: True

print(object() is object())
print(object() is not object())


# CHECK: False
# CHECK: True


def foo():
    print("foo called")
    return 2


print(1 < foo() < 3)


# CHECK-1: foo called
# CHECK: True

def bar():
    print("bar called")
    return 3


print(3 < foo() < bar())
# CHECK-NOT: bar called
# CHECK: False
