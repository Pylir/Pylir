# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

print(repr(420) == "420")
print(repr(True) == "True")
# CHECK: True
# CHECK: True

print(repr(object()))
# CHECK: <{{(.*\.)?}}object object at {{[[:alnum:]]+}}>

print(repr(None))
# CHECK: None

print(repr(NotImplemented))
# CHECK: NotImplemented
