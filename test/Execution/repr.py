# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

print(repr(420) == "420")
print(repr(True) == "True")
# CHECK: True
# CHECK: True
