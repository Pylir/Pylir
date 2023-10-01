# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

it = iter((3, 5, 6))

try:
    while True:
        print(next(it))
except StopIteration:
    pass

# CHECK: 3
# CHECK: 5
# CHECK: 6

it = iter((2, 4, 7))

while (x := next(it, None)) and x is not None:
    print(x)

# CHECK: 2
# CHECK: 4
# CHECK: 7
