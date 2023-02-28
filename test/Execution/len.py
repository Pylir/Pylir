# RUN: pylir %rt_link_flags %s -o %t -O3
# RUN: %t | FileCheck %s --match-full-lines

print(len(()))
print(len([]))
print(len({}))
print(len((3, 5)))
print(len([3, 5]))
print(len({"3": 5}))

# CHECK: 0
# CHECK: 0
# CHECK: 0
# CHECK: 2
# CHECK: 2
# CHECK: 1
