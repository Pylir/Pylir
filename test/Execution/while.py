# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

while False:
    print("Failure")
else:
    print("Success")
# CHECK-NOT: Failure
# CHECK: Success


x = 0
while True:
    if x:
        break
    x = x + 1
else:
    print("Failure")
# CHECK-NOT: Failure

x = 0
while x < 10:
    y = x
    x = x + 1
    if y < 5:
        continue
    print(y)

# CHECK: 5
# CHECK: 6
# CHECK: 7
# CHECK: 8
# CHECK: 9
