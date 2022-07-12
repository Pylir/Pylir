# RUN: pylir %s -o %t -O3
# RUN: %t | FileCheck %s --match-full-lines

try:
    raise TypeError
except TypeError:
    print("Success")
# CHECK: Success

try:
    raise TypeError("Success")
except TypeError as e:
    print(e)
# CHECK: Success
