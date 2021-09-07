# RUN: pylir %s -fsyntax-only -emit-ast 2>&1 | FileCheck %s

test
'dwadw\\nwdawdw'
b'dwadw\\nwdawdw'
5
0.5
None
True
False

# CHECK: atom test
# CHECK: atom 'dwadw\\nwdawdw'
# CHECK: atom b'dwadw\\nwdawdw'
# CHECK: atom 5
# CHECK: atom 0.5
# CHECK: atom None
# CHECK: atom True
# CHECK: atom False

()
(5)
(5,)
(5, 3)

# CHECK: parenth empty
# CHECK: parenth
# CHECK-NEXT: atom 5
# CHECK: parenth
# CHECK-NEXT: starred expression
# CHECK-NEXT: atom 5
# CHECK: parenth
# CHECK-NEXT: starred expression
# CHECK-NEXT: atom 5
# CHECK-NEXT: atom 3

(yield from 5)
(yield)
(yield 5)
(yield 5, 5,)

# CHECK: yieldatom
# CHECK-NEXT: yield from
# CHECK-NEXT: atom 5

# CHECK: yieldatom
# CHECK-NEXT: yield empty

# CHECK: yieldatom
# CHECK-NEXT: yield list
# CHECK-NEXT: atom 5

# CHECK: yieldatom
# CHECK-NEXT: yield list
# CHECK-NEXT: atom 5
# CHECK-NEXT: atom 5

[]
[5]
[5, 3]
[*5, 3]

# CHECK: list display empty
# CHECK: list display
# CHECK-NEXT: atom 5

# CHECK: list display
# CHECK-NEXT: starred list
# CHECK-NEXT: atom 5
# CHECK-NEXT: atom 3

# CHECK: list display
# CHECK-NEXT: starred list
# CHECK-NEXT: starred item
# CHECK-NEXT: atom 5
# CHECK-NEXT: atom 3

{5}
{5, 3}
{*5, 3}

# CHECK: set display
# CHECK-NEXT: atom 5

# CHECK: set display
# CHECK-NEXT: starred list
# CHECK-NEXT: atom 5
# CHECK-NEXT: atom 3

# CHECK: set display
# CHECK-NEXT: starred list
# CHECK-NEXT: starred item
# CHECK-NEXT: atom 5
# CHECK-NEXT: atom 3

{}
{5: 3}
{5: 3, 3: 2}
{**5, 3: 2}

# CHECK: dict display empty

# CHECK: dict display
# CHECK-NEXT: key
# CHECK-NEXT: atom 5
# CHECK-NEXT: atom 3

# CHECK: dict display
# CHECK-NEXT: key datum list
# CHECK-NEXT: key
# CHECK-NEXT: atom 5
# CHECK-NEXT: atom 3
# CHECK-NEXT: key
# CHECK-NEXT: atom 3
# CHECK-NEXT: atom 2

# CHECK: dict display
# CHECK-NEXT: key datum list
# CHECK-NEXT: datum
# CHECK-NEXT: atom 5
# CHECK-NEXT: key
# CHECK-NEXT: atom 3
# CHECK-NEXT: atom 2
