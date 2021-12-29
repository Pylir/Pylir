# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s
print(420)
# CHECK: 420
