
# Rationales

This document is a random document in which I am documenting design choices for the future, so that if I ever reconsider
any choices I can go over the circumstances and arguments that lead to specific decisions.

## Alloca, Load and Store

Early on the issue of how to handle variables arose. Initially I was going to emit an already pure SSA form from the 
frontend by taking care of control flow merge points and forming variant types from the incoming values. An advantage
with this approach I realised early on as well, is that trivial constant folding could occur which would automatically
also lead to type deduction. 

The reason I am currently going with alloca, load and store instead though are:

* Merging control flow points is a pain, doesn't matter whether through block arguments or structured control flow Ops,
  but ESPECIALLY using structured control flow ops. One has to modify and insert ops in the predecessors to cast to the
  variant type. The variant type itself is only known once all predecessors values have been generated
* "Casting" types is a lot easier through alloca. The alloca produces a handle type that can't be used in any ops but
  store and load. Store ops allow storing ANY kind of Python type, and loading allows to load any kind of type. The type
  erasure/casting to Pylir_UnknownType is therefore basically built in
* I am currently thinking that using a mem2reg style algorithm might be easier to implement. Type deduction should then
  work through folding as well as canonicalization patterns

## Type objects with function pointers vs Dict lookup

When I first attempted conversion the MLIRs LLVM dialect I got stuck on how to convert and handle type objects
(instances of Pythons `type` class which represents other types). They're somewhat important as the type is what
dictates which operators are overloaded.

The two designs in my mind were:

* Type objects with function pointers
* Dictionary lookup returning function objects

### Type objects with function pointers

This approach is basically the way CPython works. The `PyTypeObject` is large struct containing many function pointers
which are either null, or point the to the implementation of that method. Eg. the `tp_call` method implements `__call__`
. The builtin types initialize it to their implementation in a C function. Types defined in python will first inherit it
from the base classes (which might be builtin types). If overwritten or if the base classes `tp_call` is null, it will
assign a function to it which does the dictionary lookup for `__call__` in the object and also handles things like
descriptors and exceptions.

Pros of this approach:

* Well optimized even in absence of any optimization passes. There is no dictionary lookup required and a function call
  in Python can simply be translated to a fetch of the `tp_call` member in the `PyTypeObject` struct and then executed.
* Maybe easier compatibility with CPython. If this project is any successful and I do end up working on it for a long
  time and want to make it use able, then source compatibility with CPythons API to be able to use popular libraries
  like numpy will be required. Using this approach might make this simpler?
* Easier generation of LLVM IR
* Easier to call from the C/C++ runtime

Cons of this approach:

* ~~One has to do quite a few special cases when accessing `__dict__` as well as when setting a special method.
  Everytime I now set an attribute, which might also be on the type object, I am forced to check whether the attribute
  is `__call__` , and if it is,~~ I'd have to use a descriptor for both `__dict__` as well as the mapped value in it for
  `__call__`. Its `__set__` method must then switch `tp_call` to a generic version using dictionary lookup if
  reassigned. Same for all other special methods.

### Dictionary lookup

This approach does not treat type objects as any special and would simply make them global variables where the builtin
ones are `const` as to aid optimization and more. The type object would therefore simply be an object like any other
that has a dictionary where the key could eg. be `"__call__"` and the mapped object a callable.

Pros of this approach:

* More Generic: There is no need for special cases when setting an attribute or anything like that. Optimizations done
  to improve performance of dictionary lookup will benefit operator overloading as well and vice versa. No need to
  somehow special case `__dict__` as the function object will already be inside of it.

Cons of this approach:

* Harder interoperability, both from the runtime, and likely in the future with CPython modules.
* Harder generation of LLVM IR
* Less performant in absence of optimizations. (A use case unlikely imo, and the optimization rather trivial, but still)

### Conclusion

Going for the Dictionary lookup for now as it is much more generic. YOLO

## Representing object with known type in the type system

As of this writing Object type has an optional parameter denoting the type of the object by referring to the type
object. I've been debating whether this is the way to go or not.

My thoughts were that I had two choices:

* Analysis or transformation pass which would deduce the known type and do devirtualization
* Attach it to the type of a value and do type deduction through various passes as well

The problem was sort of, how it was supposed to be stored. Having the object type parameterized added a lot of
complexity as these were all distinct types. I introduced special call, call_indirect and return ops which in
translation to LLVM IR implicitly bitcasted object types of known type to ones of unknown type. I basically wanted to
avoid reinterpret casts as these would almost certainly turn cumbersome for passes.

I was thinking that I should simply remove the parameter of Object type as that'd be much much easier. I justified its
existence as for some operations the size of the object had to be known. With time I removed many of those operations
however. Only ones remaining are basically the allocation function.
