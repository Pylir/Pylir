# RUN: pylir %s -o %t -O3
# RUN: %t | FileCheck %s

def foo():
    a


try:
    foo()
except NameError:
    print("caught")

# CHECK: caught
