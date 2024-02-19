# RUN: pylir %s -fsyntax-only -verify

def foo():
    @pylir.intr.const_export
    # expected-error@below {{'pylir.intr.const_export' object must be defined in global scope}}
    class Bar:
        pass


@pylir.intr.const_export
# expected-error@below {{Decorators on a 'const_export' object are not supported}}
@foo
class Bar:
    pass


@pylir.intr.const_export
class Bar:
    # expected-error@below {{Decorators on a 'const_export' object are not supported}}
    @foo
    def bar(self):
        pass


@pylir.intr.const_export
# expected-error@below {{expected constant expression}}
def foo(a=[]):
    pass


@pylir.intr.const_export
# expected-error@+2 {{only positional arguments allowed in 'const_export' class inheritance list}}
# expected-error@below {{expected constant expression}}
class Foo(a=[]):
    pass


@pylir.intr.const_export
# expected-error@+2 {{only positional arguments allowed in 'const_export' class inheritance list}}
# expected-error@below {{expected constant expression}}
class Foo(*[]):
    pass


@pylir.intr.const_export
# expected-error@below {{expected constant expression}}
class Foo([]):
    pass
