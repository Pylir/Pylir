# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines
print(420)
# CHECK: 420

print(str(420) == "420")
# CHECK: True
