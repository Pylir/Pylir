# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s
print("Hello World!")
# CHECK: Hello World!
