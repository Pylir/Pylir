# RUN: pylir %s -fsyntax-only -dump-ast | FileCheck --match-full-lines %s

a.b
# CHECK: |-attribute b
# CHECK-NEXT: | `-atom a

a[b]

# CHECK-NEXT: |-subscription
# CHECK-NEXT: | |-object: atom a
# CHECK-NEXT: | `-index: atom b

a[b,c]

# CHECK-NEXT: |-subscription
# CHECK-NEXT: | |-object: atom a
# CHECK-NEXT: | `-index: tuple construct
# CHECK-NEXT: |   |-atom b
# CHECK-NEXT: |   `-atom c

a[b::]

# CHECK-NEXT: |-subscription
# CHECK-NEXT: | |-object: atom a
# CHECK-NEXT: | `-index: slice
# CHECK-NEXT: |   `-lowerBound: atom b

a[:c:]

# CHECK-NEXT: |-subscription
# CHECK-NEXT: | |-object: atom a
# CHECK-NEXT: | `-index: slice
# CHECK-NEXT: |   `-upperBound: atom c

a[::c]

# CHECK-NEXT: |-subscription
# CHECK-NEXT: | |-object: atom a
# CHECK-NEXT: | `-index: slice
# CHECK-NEXT: |   `-stride: atom c

a[b:c:d]

# CHECK-NEXT: |-subscription
# CHECK-NEXT: | |-object: atom a
# CHECK-NEXT: | `-index: slice
# CHECK-NEXT: |   |-lowerBound: atom b
# CHECK-NEXT: |   |-upperBound: atom c
# CHECK-NEXT: |   `-stride: atom d

a[b:c:d,3]

# CHECK-NEXT: |-subscription
# CHECK-NEXT: | |-object: atom a
# CHECK-NEXT: | `-index: tuple construct
# CHECK-NEXT: |   |-slice
# CHECK-NEXT: |   | |-lowerBound: atom b
# CHECK-NEXT: |   | |-upperBound: atom c
# CHECK-NEXT: |   | `-stride: atom d
# CHECK-NEXT: |   `-atom 3

a()
# CHECK-NEXT: |-call
# CHECK-NEXT: | `-callable: atom a

a(b)
# CHECK-NEXT: |-call
# CHECK-NEXT: | |-callable: atom a
# CHECK-NEXT: | `-argument
# CHECK-NEXT: |   `-atom b

a(b,c)
# CHECK-NEXT: |-call
# CHECK-NEXT: | |-callable: atom a
# CHECK-NEXT: | |-argument
# CHECK-NEXT: | | `-atom b
# CHECK-NEXT: | `-argument
# CHECK-NEXT: |   `-atom c

a(b,*c)
# CHECK-NEXT: |-call
# CHECK-NEXT: | |-callable: atom a
# CHECK-NEXT: | |-argument
# CHECK-NEXT: | | `-atom b
# CHECK-NEXT: | `-argument
# CHECK-NEXT: |   `-starred
# CHECK-NEXT: |     `-atom c

a(b,c = 3)
# CHECK-NEXT: |-call
# CHECK-NEXT: | |-callable: atom a
# CHECK-NEXT: | |-argument
# CHECK-NEXT: | | `-atom b
# CHECK-NEXT: | `-argument
# CHECK-NEXT: |   `-keyword item c
# CHECK-NEXT: |     `-atom 3

a(b,c = 3, *b)
# CHECK-NEXT: |-call
# CHECK-NEXT: | |-callable: atom a
# CHECK-NEXT: | |-argument
# CHECK-NEXT: | | `-atom b
# CHECK-NEXT: | |-argument
# CHECK-NEXT: | | `-keyword item c
# CHECK-NEXT: | |   `-atom 3
# CHECK-NEXT: | `-argument
# CHECK-NEXT: |   `-starred
# CHECK-NEXT: |     `-atom b

a(**b,c = 3)
# CHECK-NEXT: |-call
# CHECK-NEXT: | |-callable: atom a
# CHECK-NEXT: | |-argument
# CHECK-NEXT: | | `-mapped
# CHECK-NEXT: | |   `-atom b
# CHECK-NEXT: | `-argument
# CHECK-NEXT: |   `-keyword item c
# CHECK-NEXT: |     `-atom 3

pylir.intr.const_export
# CHECK-NEXT: |-intrinsic pylir.intr.const_export
pylir.intr.const_export()
# CHECK-NEXT: |-call
# CHECK-NEXT: | `-callable: intrinsic pylir.intr.const_export
pylir.intr.const_export[0]
# CHECK-NEXT: `-subscription
# CHECK-NEXT:   |-object: intrinsic pylir.intr.const_export
# CHECK-NEXT:   `-index: atom 0
