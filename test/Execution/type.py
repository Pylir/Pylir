# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

print(type(0) == int)
# CHECK: True
