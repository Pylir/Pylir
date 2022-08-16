# RUN: pylir %s -o %t -O3
# RUN: %t | FileCheck %s --match-full-lines

i = 5
i += 5
print(i)
# CHECK: 10
