# RUN: pylir %rt_link_flags %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

print(type(0) == int)
# CHECK: True
