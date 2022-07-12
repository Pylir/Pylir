# RUN: pylir %s -o %t -O3
# RUN: %t | FileCheck %s --match-full-lines

print(type(0) == int)
# CHECK: True
