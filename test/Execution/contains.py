# RUN: pylir %rt_link_flags %s -o %t -O3
# RUN: %t | FileCheck %s --match-full-lines

d = {"one": 1, "two": 2, "five": 5}
print("one" in d)
print("two" in d)
print("three" in d)
print("four" in d)
print("five" in d)

# CHECK: True
# CHECK: True
# CHECK: False
# CHECK: False
# CHECK: True
