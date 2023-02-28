# RUN: pylir %rt_link_flags %s -o %t -O3
# RUN: %t | FileCheck %s --match-full-lines

print("Hello" + " " + "World")
# CHECK: Hello World
print(str())
# CHECK-EMPTY:

print(str("Text"))
# CHECK: Text

print("Hello" == "Hello")
# CHECK: True
