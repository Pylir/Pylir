# RUN: pylir %s -fsyntax-only -dump-ast | FileCheck %s --match-full-lines

test
'dwadw\\nwdawdw'
b'dwadw\\nwdawdw'
5
0.5
None
True
False

# CHECK: |-atom test
# CHECK-NEXT: |-atom 'dwadw\\nwdawdw'
# CHECK-NEXT: |-atom b'dwadw\\nwdawdw'
# CHECK-NEXT: |-atom 5
# CHECK-NEXT: |-atom 0.5
# CHECK-NEXT: |-atom None
# CHECK-NEXT: |-atom True
# CHECK-NEXT: |-atom False

()
(5)
(5,)
(5, 3)

# CHECK-NEXT: |-tuple construct
# CHECK-NEXT: |-atom 5
# CHECK-NEXT: |-tuple construct
# CHECK-NEXT: | `-atom 5
# CHECK-NEXT: |-tuple construct
# CHECK-NEXT: | |-atom 5
# CHECK-NEXT: | `-atom 3

def foo():
    (yield from 5)
    (yield)
    (yield 5)
    (yield 5, 5,)

# CHECK:      |   |-yield from
# CHECK-NEXT: |   | `-atom 5
# CHECK-NEXT: |   |-yield empty
# CHECK-NEXT: |   |-yield
# CHECK-NEXT: |   | `-atom 5
# CHECK-NEXT: |   `-yield
# CHECK-NEXT: |     `-tuple construct
# CHECK-NEXT: |       |-atom 5
# CHECK-NEXT: |       `-atom 5

[]
[5]
[5, 3]
[*5, 3]
[*5]

# CHECK-NEXT: |-list display
# CHECK-NEXT: |-list display
# CHECK-NEXT: | `-atom 5
# CHECK-NEXT: |-list display
# CHECK-NEXT: | |-atom 5
# CHECK-NEXT: | `-atom 3
# CHECK-NEXT: |-list display
# CHECK-NEXT: | |-starred item
# CHECK-NEXT: | | `-atom 5
# CHECK-NEXT: | `-atom 3
# CHECK-NEXT: |-list display
# CHECK-NEXT: | `-starred item
# CHECK-NEXT: |   `-atom 5

{5}
{5, 3}
{*5, 3}
{*5}

# CHECK-NEXT: |-set display
# CHECK-NEXT: | `-atom 5
# CHECK-NEXT: |-set display
# CHECK-NEXT: | |-atom 5
# CHECK-NEXT: | `-atom 3
# CHECK-NEXT: |-set display
# CHECK-NEXT: | |-starred item
# CHECK-NEXT: | | `-atom 5
# CHECK-NEXT: | `-atom 3
# CHECK-NEXT: |-set display
# CHECK-NEXT: | `-starred item
# CHECK-NEXT: |   `-atom 5


{}
{5: 3}
{5: 3, 3: 2}
{**5, 3: 2}

# CHECK-NEXT: |-dict display
# CHECK-NEXT: |-dict display
# CHECK-NEXT: | |-key: atom 5
# CHECK-NEXT: | `-value: atom 3
# CHECK-NEXT: |-dict display
# CHECK-NEXT: | |-key: atom 5
# CHECK-NEXT: | |-value: atom 3
# CHECK-NEXT: | |-key: atom 3
# CHECK-NEXT: | `-value: atom 2
# CHECK-NEXT: `-dict display
# CHECK-NEXT:   |-unpacked: atom 5
# CHECK-NEXT:   |-key: atom 3
# CHECK-NEXT:   `-value: atom 2
