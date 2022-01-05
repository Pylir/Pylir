# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

if True:
    print("Success")
# CHECK: Success

if False:
    print("Failure")
# CHECK-NOT: Failure
