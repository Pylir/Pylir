# RUN: pylir %s -emit-pylir -o - -S -verify

# expected-error@below {{unknown intrinsic 'pylir.intr.thing.doesnt.exist'}}
pylir.intr.thing.doesnt.exist()

args = 1
d = 2

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

