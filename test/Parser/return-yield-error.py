# RUN: pylir %s -fsyntax-only -verify

# expected-error@below {{occurrence of 'return' outside of function}}
return


class Foo:
    # expected-error@below {{occurrence of 'return' outside of function}}
    return


def foo():
    class Foo:
        # expected-error@below {{occurrence of 'return' outside of function}}
        return

# expected-error@below {{occurrence of 'yield' outside of function}}
yield 5


class Foo:
    # expected-error@below {{occurrence of 'yield' outside of function}}
    yield 5


def foo():
    class Foo:
        # expected-error@below {{occurrence of 'yield' outside of function}}
        yield 5

