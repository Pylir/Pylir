# RUN: pylir %s -emit-pylir -o - -S -verify

# expected-error@below {{unknown intrinsic 'pylir.intr.thing.doesnt.exist'}}
pylir.intr.thing.doesnt.exist

# expected-error@below {{unknown intrinsic 'pylir.intr.thing.doesnt.exist'}}
pylir.intr.thing.doesnt.exist()

args = ()
d = {}

# expected-error@+3 {{intrinsics do not support keyword arguments}}
# expected-error@+2 {{intrinsics do not support iterable unpacking arguments}}
# expected-error@+1 {{intrinsics do not support dictionary unpacking arguments}}
pylir.intr.typeOf(k=3, *args, **d)

# expected-error@below {{intrinsics do not support comprehension arguments}}
pylir.intr.typeOf(i for i in d)

# expected-error@below {{intrinsic 'pylir.intr.typeOf' expects 1 argument(s) not 0}}
pylir.intr.typeOf()

# expected-error@below {{argument 1 of intrinsic 'pylir.intr.int.cmp' has to be a constant string}}
pylir.intr.int.cmp(args, d, args)

# expected-error@below {{invalid enum value 'lol' for enum 'IntCmpKind' argument}}
pylir.intr.int.cmp('lol', d, args)
# expected-note@above {{valid values are: eq, ne, lt, le, gt, ge}}


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
