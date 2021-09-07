# RUN: pylir %s -fsyntax-only -emit-ast 2>&1 | FileCheck %s

await 5

# CHECK: await expression
# CHECK-NEXT: atom 5

-5

# CHECK: unary '-'
# CHECK-NEXT: atom 5

+5

# CHECK: unary '+'
# CHECK-NEXT: atom 5

~5

# CHECK: unary '~'
# CHECK-NEXT: atom 5

+-5

# CHECK: unary '+'
# CHECK-NEXT: unary '-'
# CHECK-NEXT: atom 5

2 ** 5
# CHECK: power
# CHECK-NEXT: base: atom 2
# CHECK-NEXT: exponent: atom 5

await 2 ** 5
# CHECK: power
# CHECK-NEXT: base: await expression
# CHECK-NEXT: atom 2
# CHECK-NEXT: exponent: atom 5

2 * 5
# CHECK: mexpr '*'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 @ 5
# CHECK: mexpr '@'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 // 5
# CHECK: mexpr '//'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 / 5
# CHECK: mexpr '/'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 % 5
# CHECK: mexpr '%'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 @ 5 * 3
# CHECK: mexpr '@'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: mexpr '*'
# CHECK-NEXT: lhs: atom 5
# CHECK-NEXT: rhs: atom 3

2 / 5 * 3
# CHECK: mexpr '*'
# CHECK-NEXT: lhs: mexpr '/'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5
# CHECK-NEXT: rhs: atom 3

2 + 5
# CHECK: aexpr '+'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 - 5
# CHECK: aexpr '-'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 << 5
# CHECK: shiftExpr '<<'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 >> 5
# CHECK: shiftExpr '>>'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 & 5
# CHECK: andExpr '&'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 ^ 5
# CHECK: xorExpr '^'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 | 5
# CHECK: orExpr '|'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 < 5
# CHECK: comparison
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: '<': atom 5

2 > 5
# CHECK: comparison
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: '>': atom 5

2 <= 5
# CHECK: comparison
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: '<=': atom 5

2 >= 5
# CHECK: comparison
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: '>=': atom 5

2 == 5
# CHECK: comparison
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: '==': atom 5

2 != 5
# CHECK: comparison
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: '!=': atom 5

2 is 5
# CHECK: comparison
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: 'is': atom 5

2 is not 5
# CHECK: comparison
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: 'is' 'not': atom 5

2 in 5
# CHECK: comparison
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: 'in': atom 5

2 not in 5
# CHECK: comparison
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: 'not' 'in': atom 5

not 2
# CHECK: notTest
# CHECK-NEXT: atom 2

not not 2
# CHECK: notTest
# CHECK-NEXT: notTest
# CHECK-NEXT: atom 2

2 or 5
# CHECK: orTest 'or'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 and 5
# CHECK: andTest 'and'
# CHECK-NEXT: lhs: atom 2
# CHECK-NEXT: rhs: atom 5

2 if 3 else 5
# CHECK: conditional expression
# CHECK-NEXT: value: atom 2
# CHECK-NEXT: condition: atom 3
# CHECK-NEXT: elseValue: atom 5

lambda: 3
# CHECK: lambda expression
# CHECK-NEXT: atom 3

a = 3
# CHECK: assignment statement
# CHECK-NEXT: target a
# CHECK-NEXT: atom 3

[a] = 3
# CHECK: assignment statement
# CHECK-NEXT: target square
# CHECK-NEXT: target a
# CHECK-NEXT: atom 3

a += 3
# CHECK: augmented assignment '+='
# CHECK-NEXT: augtarget a
# CHECK-NEXT: atom 3

a: b
# CHECK: annotated assignment
# CHECK-NEXT: augtarget a
# CHECK-NEXT: atom b


a: b = 3
# CHECK: annotated assignment
# CHECK-NEXT: augtarget a
# CHECK-NEXT: atom b
# CHECK-NEXT: atom 3

assert 5
# CHECK: assert statement
# CHECK-NEXT: condition: atom 5

assert 5, 3
# CHECK: assert statement
# CHECK-NEXT: condition: atom 5
# CHECK-NEXT: message: atom 3

pass
# CHECK: pass statement

while True:
    continue
# CHECK: continue statement

while True:
    break
# CHECK: break statement

del a


# CHECK: del statement
# CHECK-NEXT: target a

def foo():
    return


# CHECK: return statement

def foo():
    return 5


# CHECK: return statement
# CHECK: atom 5

raise
# CHECK: raise statement

raise 5
# CHECK: raise statement
# CHECK: exception: atom 5

raise 5 from 3
# CHECK: raise statement
# CHECK: exception: atom 5
# CHECK: expression: atom 3

global a, b, c
# CHECK: global a, b, c

nonlocal a, b, c
# CHECK: nonlocal a, b, c

a = 3; b = 4
# CHECK: stmt list
# CHECK-NEXT: assignment statement
# CHECK-NEXT: target a
# CHECK-NEXT: atom 3
# CHECK-NEXT: assignment statement
# CHECK-NEXT: target b
# CHECK-NEXT: atom 4
