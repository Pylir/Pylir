# RUN: pylir %rt_link_flags %s -o %t
# RUN: %t | FileCheck %s
print("Hello World!")
# CHECK: Hello World!
