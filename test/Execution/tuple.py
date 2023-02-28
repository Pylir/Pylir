# RUN: pylir %rt_link_flags %s -o %t -O3
# RUN: %t | FileCheck %s --match-full-lines

print((3, 353))
# CHECK: (3, 353)

print(())
# CHECK: ()

print((3,))
# CHECK: (3,)
