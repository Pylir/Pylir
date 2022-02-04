# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

print(True)
# CHECK: True
print(False)
# CHECK: False

print(str(True) == "True")
# CHECK: True

print(str(False) == "False")
# CHECK: True

print(bool())
# CHECK: False

print(bool(False))
# CHECK: False

print(bool(True))
# CHECK: True

print(bool(None))
# CHECK: False

# Calls len afterwards

print(bool([]))
# CHECK: False

print(bool([0]))
# CHECK: True

print(bool(object()))
# CHECK: True

print(True + True)
# CHECK: 2

print(False + True)
# CHECK: 1
