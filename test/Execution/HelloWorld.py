# RUN: pylir %s -o %t -O3
# RUN: %t | FileCheck %s
print("Hello World!")
# CHECK: Hello World!
