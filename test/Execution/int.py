# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines
print(420)
# CHECK: 420

print(str(420) == "420")
# CHECK: True

if 0:
    print("Failure")
# CHECK-NOT: Failure

if 1:
    print("Success")
# CHECK: Success

print(5 + 7)
# CHECK: 12
