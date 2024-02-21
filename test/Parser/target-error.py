# RUN: pylir %s -fsyntax-only -verify

# expected-error@below {{cannot assign to result of operator ':='}}
(a := 3) = 3

# expected-error@below {{cannot assign to result of lambda expression}}
(lambda: 3) = 3

# expected-error@below {{cannot assign to result of conditional expression}}
(3 if True else 5) = 3

# expected-error@below {{cannot assign to result of operator 'and'}}
(3 and 5) = 3

# expected-error@below {{cannot assign to result of unary operator 'not'}}
(not 3) = 3

# expected-error@below {{cannot assign to result of operator '!='}}
(3 != 5) = 3

# expected-error@below {{cannot assign to result of unary operator '-'}}
(-3) = 3

# expected-error@below {{cannot assign to result of operator '**'}}
(2 ** 8) = 3

# expected-error@below {{cannot assign to result of unary operator 'await'}}
(await foo()) = 3

# expected-error@below {{cannot assign to result of call}}
(foo()) = 3

# expected-error@below {{cannot assign to literal}}
5 = 3

# expected-error@below {{cannot assign to dictionary}}
{} = 3

# expected-error@below {{cannot assign to set}}
{5} = 3

# expected-error@below {{cannot assign to list comprehension}}
[5 for c in f] = 3

# expected-error@below {{cannot assign to literal}}
[5] = 3


# expected-error@below {{cannot assign to yield expression}}
def foo(): (yield 5) = 3


# expected-error@below {{cannot assign to generator expression}}
(c for c in f) = 3

# expected-error@below {{operator '+=' cannot assign to multiple variables}}
a, b += 3

# expected-error@below {{cannot delete iterable unpacking}}
del (*a,)

# expected-error@below {{cannot delete iterable unpacking}}
del [*a]

# expected-error@below {{only one iterable unpacking possible in assignment}}
*a, b, *c = 3
# expected-note@above {{previous occurrence here}}

# expected-error@below {{cannot assign to intrinsic}}
pylir.intr.const_export = 3
