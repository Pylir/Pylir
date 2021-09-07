# RUN: pylir %s -fsyntax-only -emit-ast 2>&1 | FileCheck %s

a.b

# CHECK: attribute b
# CHECK-NEXT: atom a

a[b]

# CHECK: subscription
# CHECK-NEXT: primary: atom a
# CHECK-NEXT: index: atom b

a[b,c]

# CHECK: subscription
# CHECK-NEXT: primary: atom a
# CHECK-NEXT: index: expression list
# CHECK-NEXT: atom b
# CHECK-NEXT: atom c

a[b::]

# CHECK: slicing
# CHECK-NEXT: primary: atom a
# CHECK-NEXT: index: proper slice
# CHECK-NEXT: lowerBound: atom b

a[:c:]

# CHECK: slicing
# CHECK-NEXT: primary: atom a
# CHECK-NEXT: index: proper slice
# CHECK-NEXT: upperBound: atom c

a[::c]

# CHECK: slicing
# CHECK-NEXT: primary: atom a
# CHECK-NEXT: index: proper slice
# CHECK-NEXT: stride: atom c

a[b:c:d]

# CHECK: slicing
# CHECK-NEXT: primary: atom a
# CHECK-NEXT: index: proper slice
# CHECK-NEXT: lowerBound: atom b
# CHECK-NEXT: upperBound: atom c
# CHECK-NEXT: stride: atom d

a[b:c:d,3]

# CHECK: slicing
# CHECK-NEXT: primary: atom a
# CHECK-NEXT: index: proper slice list
# CHECK-NEXT: proper slice
# CHECK-NEXT: lowerBound: atom b
# CHECK-NEXT: upperBound: atom c
# CHECK-NEXT: stride: atom d
# CHECK-NEXT: atom 3

a()
# CHECK: call
# CHECK-NEXT: atom a

a(b)
# CHECK: call
# CHECK-NEXT: atom a
# CHECK-NEXT: argument list
# CHECK-NEXT: positional arguments
# CHECK-NEXT: atom b

a(b,c)
# CHECK: call
# CHECK-NEXT: atom a
# CHECK-NEXT: argument list
# CHECK-NEXT: positional arguments
# CHECK-NEXT: atom b
# CHECK-NEXT: atom c

a(b,*c)
# CHECK: call
# CHECK-NEXT: atom a
# CHECK-NEXT: argument list
# CHECK-NEXT: positional arguments
# CHECK-NEXT: atom b
# CHECK-NEXT: starred
# CHECK-NEXT: atom c

a(b,c = 3)
# CHECK: call
# CHECK-NEXT: atom a
# CHECK-NEXT: argument list
# CHECK-NEXT: positional arguments
# CHECK-NEXT: atom b
# CHECK-NEXT: starred keywords
# CHECK-NEXT: keyword item c
# CHECK-NEXT: atom 3

a(b,c = 3, *b)
# CHECK: call
# CHECK-NEXT: atom a
# CHECK-NEXT: argument list
# CHECK-NEXT: positional arguments
# CHECK-NEXT: atom b
# CHECK-NEXT: starred keywords
# CHECK-NEXT: keyword item c
# CHECK-NEXT: atom 3
# CHECK-NEXT: starred expression
# CHECK-NEXT: atom b

a(**b,c = 3)
# CHECK: call
# CHECK-NEXT: atom a
# CHECK-NEXT: argument list
# CHECK-NEXT: keyword arguments
# CHECK-NEXT: mapped expression
# CHECK-NEXT: atom b
# CHECK-NEXT: keyword item c
# CHECK-NEXT: atom 3
