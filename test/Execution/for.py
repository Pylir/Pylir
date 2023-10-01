# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

for i in (3, 5, 6):
    print(i)
# CHECK: 3
# CHECK: 5
# CHECK: 6


for i in (3, 5, 6):
    break
else:
    print("Failure")

# CHECK-NOT: Failure

for i in (3, 5, 6):
    continue
else:
    print(i)

# CHECK: 6
