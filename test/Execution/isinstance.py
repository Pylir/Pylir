# RUN: pylir %rt_link_flags %s -o %t -O3
# RUN: %t | FileCheck %s --match-full-lines

print(isinstance(5, int))
# CHECK: True
print(isinstance(True, int))
# CHECK: True
print(isinstance(True, str))
# CHECK: False
print(isinstance(5, bool))
# CHECK: False
print(isinstance(True, (str, int)))
# CHECK: True
print(isinstance((), (str, type)))
# CHECK: False
