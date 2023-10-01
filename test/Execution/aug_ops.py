# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

i = 5
i += 5
print(i)
# CHECK: 10
