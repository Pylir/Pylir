# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s

class Test:
    x = 3


t = Test()
# CHECK: <__main__.Test object at {{[[:alnum:]]+}}>
print(t)

# CHECK: True
print(Test.x == 3)
# CHECK: True
print(t.x == 3)

t.x = 5
# CHECK: True
print(Test.x == 3)
# CHECK: True
print(t.x == 5)


class Test:
    __slots__ = ('x', )


t = Test()
try:
    t.x = 3
except AttributeError:
    # CHECK: Success
    print('Success')
