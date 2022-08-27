# RUN: pylir %s -fsyntax-only -dump-ast | FileCheck --match-full-lines %s

await 5

# CHECK:      |-unary op 'await'
# CHECK-NEXT: | `-atom 5

-5

# CHECK-NEXT: |-unary op '-'
# CHECK-NEXT: | `-atom 5

+5

# CHECK-NEXT: |-unary op '+'
# CHECK-NEXT: | `-atom 5

~5

# CHECK-NEXT: |-unary op '~'
# CHECK-NEXT: | `-atom 5

+-5

# CHECK-NEXT: |-unary op '+'
# CHECK-NEXT: | `-unary op '-'
# CHECK-NEXT: |   `-atom 5

2 ** 5
# CHECK-NEXT: |-binary op '**'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

await 2 ** 5
# CHECK-NEXT: |-binary op '**'
# CHECK-NEXT: | |-unary op 'await'
# CHECK-NEXT: | | `-atom 2
# CHECK-NEXT: | `-atom 5

2 * 5
# CHECK-NEXT: |-binary op '*'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 @ 5
# CHECK-NEXT: |-binary op '@'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 // 5
# CHECK-NEXT: |-binary op '//'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 / 5
# CHECK-NEXT: |-binary op '/'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 % 5
# CHECK-NEXT: |-binary op '%'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 @ 5 * 3
# CHECK-NEXT: |-binary op '@'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-binary op '*'
# CHECK-NEXT: |   |-atom 5
# CHECK-NEXT: |   `-atom 3

2 / 5 * 3
# CHECK-NEXT: |-binary op '*'
# CHECK-NEXT: | |-binary op '/'
# CHECK-NEXT: | | |-atom 2
# CHECK-NEXT: | | `-atom 5
# CHECK-NEXT: | `-atom 3

2 + 5
# CHECK-NEXT: |-binary op '+'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 - 5
# CHECK-NEXT: |-binary op '-'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 << 5
# CHECK-NEXT: |-binary op '<<'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 >> 5
# CHECK-NEXT: |-binary op '>>'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 & 5
# CHECK-NEXT: |-binary op '&'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 ^ 5
# CHECK-NEXT: |-binary op '^'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 | 5
# CHECK-NEXT: |-binary op '|'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 < 5
# CHECK-NEXT: |-comparison
# CHECK-NEXT: | |-lhs: atom 2
# CHECK-NEXT: | `-'<': atom 5

2 > 5
# CHECK-NEXT: |-comparison
# CHECK-NEXT: | |-lhs: atom 2
# CHECK-NEXT: | `-'>': atom 5

2 <= 5
# CHECK-NEXT: |-comparison
# CHECK-NEXT: | |-lhs: atom 2
# CHECK-NEXT: | `-'<=': atom 5

2 >= 5
# CHECK-NEXT: |-comparison
# CHECK-NEXT: | |-lhs: atom 2
# CHECK-NEXT: | `-'>=': atom 5

2 == 5
# CHECK-NEXT: |-comparison
# CHECK-NEXT: | |-lhs: atom 2
# CHECK-NEXT: | `-'==': atom 5

2 != 5
# CHECK-NEXT: |-comparison
# CHECK-NEXT: | |-lhs: atom 2
# CHECK-NEXT: | `-'!=': atom 5

2 is 5
# CHECK-NEXT: |-comparison
# CHECK-NEXT: | |-lhs: atom 2
# CHECK-NEXT: | `-'is': atom 5

2 is not 5
# CHECK-NEXT: |-comparison
# CHECK-NEXT: | |-lhs: atom 2
# CHECK-NEXT: | `-'is' 'not': atom 5

2 in 5
# CHECK-NEXT: |-comparison
# CHECK-NEXT: | |-lhs: atom 2
# CHECK-NEXT: | `-'in': atom 5

2 not in 5
# CHECK-NEXT: |-comparison
# CHECK-NEXT: | |-lhs: atom 2
# CHECK-NEXT: | `-'not' 'in': atom 5

not 2
# CHECK-NEXT: |-unary op 'not'
# CHECK-NEXT: | `-atom 2

not not 2
# CHECK-NEXT: |-unary op 'not'
# CHECK-NEXT: | `-unary op 'not'
# CHECK-NEXT: |   `-atom 2

2 or 5
# CHECK-NEXT: |-binary op 'or'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 and 5
# CHECK-NEXT: |-binary op 'and'
# CHECK-NEXT: | |-atom 2
# CHECK-NEXT: | `-atom 5

2 if 3 else 5
# CHECK-NEXT: |-conditional expression
# CHECK-NEXT: | |-trueValue: atom 2
# CHECK-NEXT: | |-condition: atom 3
# CHECK-NEXT: | `-elseValue: atom 5

lambda: 3
# CHECK-NEXT: |-lambda expression
# CHECK-NEXT: | `-atom 3

a = 3
# CHECK-NEXT: |-assignment statement
# CHECK-NEXT: | |-Operator '=': atom a
# CHECK-NEXT: | `-expression: atom 3

[a] = 3
# CHECK-NEXT: |-assignment statement
# CHECK-NEXT: | |-Operator '=': list display
# CHECK-NEXT: | | `-atom a
# CHECK-NEXT: | `-expression: atom 3

a += 3
# CHECK-NEXT: |-assignment statement
# CHECK-NEXT: | |-Operator '+=': atom a
# CHECK-NEXT: | `-expression: atom 3

a: b
# CHECK-NEXT: |-assignment statement
# CHECK-NEXT: | |-Operator ':': atom a
# CHECK-NEXT: | `-annotation: atom b


a: b = 3
# CHECK-NEXT: |-assignment statement
# CHECK-NEXT: | |-Operator ':': atom a
# CHECK-NEXT: | |-annotation: atom b
# CHECK-NEXT: | `-expression: atom 3

assert 5
# CHECK-NEXT: |-assert statement
# CHECK-NEXT: | `-condition: atom 5

assert 5, 3
# CHECK-NEXT: |-assert statement
# CHECK-NEXT: | |-condition: atom 5
# CHECK-NEXT: | `-message: atom 3

pass
# CHECK-NEXT: |-'pass' statement

while True:
    continue
# CHECK: |   `-'continue' statement

while True:
    break
# CHECK: |   `-'break' statement

del a
# CHECK-NEXT: |-del statement
# CHECK-NEXT: | `-atom a

def foo():
    return


# CHECK: |   `-return statement

def foo():
    return 5


# CHECK:      |   `-return statement
# CHECK-NEXT: |     `-atom 5

raise
# CHECK-NEXT: |-raise statement

raise 5
# CHECK-NEXT: |-raise statement
# CHECK-NEXT: | `-exception: atom 5

raise 5 from 3
# CHECK-NEXT: |-raise statement
# CHECK-NEXT: | |-exception: atom 5
# CHECK-NEXT: | `-cause: atom 3

global a, b, c
# CHECK-NEXT: |-global a, b, c

def foo():
    a = b = c = 3

    def outer():
        nonlocal a, b, c


# CHECK: | `-nonlocal a, b, c

a = 3; b = 4
# CHECK-NEXT: |-assignment statement
# CHECK-NEXT: | |-Operator '=': atom a
# CHECK-NEXT: | `-expression: atom 3
# CHECK-NEXT: `-assignment statement
# CHECK-NEXT: |-Operator '=': atom b
# CHECK-NEXT: `-expression: atom 4
