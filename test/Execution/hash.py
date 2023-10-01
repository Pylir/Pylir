# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

print(hash(5) == hash(5))
# CHECK: True
print(hash("text") == hash("text"))
# CHECK: True

o = object()
print(hash(o) == hash(o))
# CHECK: True
