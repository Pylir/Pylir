# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

print(True)
# CHECK: True

print(False)
# CHECK: False
