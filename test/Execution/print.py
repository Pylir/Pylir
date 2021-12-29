# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines
print("text")
# CHECK: text

print("multiple", "arguments")
# CHECK: multiple arguments

print("multiple", "arguments", sep=";")
# CHECK: multiple;arguments

print("converts", 5)
# CHECK: converts 5

print("text", end="!\n")
# CHECK: text!

print("text", sep=None, end=None)
# CHECK: text
