# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s

class Test:
    pass


t = Test()
# CHECK: <__main__.Test object at {{[[:alnum:]]+}}>
print(t)
