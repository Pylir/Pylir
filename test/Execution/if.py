# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

if True:
    print("Success")
# CHECK: Success

if False:
    print("Failure")
# CHECK-NOT: Failure

if [0]:
    print("Success")
# CHECK: Success

if []:
    print("Failure")
# CHECK-NOT: Failure

if object():
    print("Success")
# CHECK: Success

if True:
    print("Success")
else:
    print("Failure")
# CHECK: Success


if False:
    print("Failure")
elif True:
    print("Success")
# CHECK: Success

if False:
    print("Failure")
elif False:
    print("Failure")
# CHECK-NOT: Failure

if False:
    print("Failure")
elif False:
    print("Failure")
else:
    print("Success")
# CHECK: Success
