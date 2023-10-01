# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

obj = object()

try:
    print(obj.thing)
    print("Failure")
except AttributeError:
    print("Success")

# CHECK: Success

try:
    obj.thing = 3
    print("Failure")
except AttributeError:
    print("Success")

# CHECK: Success
