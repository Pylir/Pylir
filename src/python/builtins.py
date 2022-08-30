#  // Copyright 2022 Markus Böck
#  //
#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pylir.intr.BaseException
import pylir.intr.bool
import pylir.intr.dict
import pylir.intr.function
import pylir.intr.int
import pylir.intr.intr
import pylir.intr.list
import pylir.intr.object
import pylir.intr.str
import pylir.intr.tuple
import pylir.intr.type


# TODO: replace with more generic method_call once we have proper iter and
#       dictionary unpacking
def unary_method_call(method, obj):
    mro = pylir.intr.type.mro(type(method))
    t = pylir.intr.mroLookup(mro, "__get__")
    if t[1]:
        method = t[0](obj, type(obj))
    # TODO: This is incorrect. One should not be passing any arguments, and obj
    #       should be bound as the self parameter by the __get__ implementation
    #       of function. Until this is implemented we are doing this
    #       incorrectly.
    return method(obj)


def binary_method_call(method, self, other):
    mro = pylir.intr.type.mro(type(method))
    t = pylir.intr.mroLookup(mro, "__get__")
    if t[1]:
        method = t[0](self, type(self))
    # TODO: This is incorrect; See unary_method call description for details.
    #       Should be `method(other)` in the future.
    return method(self, other)


@pylir.intr.const_export
class type:
    __slots__ = pylir.intr.type.__slots__

    def __call__(self, *args, **kwargs):
        # I usually try to avoid intrinsics where possible to have as much of a
        # readable and normal python code as possible but due to special
        # importance of the below code path, this code is somewhat exempt.
        # The short of it is, that calling 'len' or using the '[]' binary
        # operator would lead to an infinite recursion in this case as all
        # these implementations call 'type'. I don't want to forbid that either
        # and hence we'll use intrinsics here that do not touch any other
        # builtin types, functions or operators.
        tuple_len = pylir.intr.tuple.len(args)
        dict_len = pylir.intr.dict.len(kwargs)
        tuple_len_is_one = pylir.intr.int.cmp("eq", tuple_len, 1)
        dict_len_is_zero = pylir.intr.int.cmp("eq", dict_len, 0)
        if self is type and tuple_len_is_one and dict_len_is_zero:
            return pylir.intr.typeOf(pylir.intr.tuple.getItem(args, 0))

        mro = pylir.intr.type.mro(self)
        new_method = pylir.intr.tuple.getItem(
            pylir.intr.mroLookup(mro, "__new__"), 0)
        # TODO: This is not the proper sequence to call a function and only
        #       works for `function`, but not arbitrary callables.
        #       Proper would be 'new_method(new_method,*args,**kwargs)' once
        #       unpacking is supported.
        res = pylir.intr.function.call(new_method, new_method,
                                       pylir.intr.tuple.prepend(self, args),
                                       kwargs)
        res_type = pylir.intr.typeOf(res)
        mro = pylir.intr.type.mro(res_type)
        if not pylir.intr.tuple.contains(mro, self):
            return res
        init_method = pylir.intr.tuple.getItem(
            pylir.intr.mroLookup(mro, "__init__"), 0)
        # TODO: Same as above but:
        #       'init_method(init_method, res, *args, **kwargs)'
        init_ret = pylir.intr.function.call(init_method, init_method,
                                            pylir.intr.tuple.prepend(res, args),
                                            kwargs)
        if init_ret is not None:
            raise TypeError
        return res


@pylir.intr.const_export
class object:
    def __new__(cls, *args, **kwargs):
        return pylir.intr.makeObject(cls)

    def __init__(self):
        pass

    def __eq__(self, other):
        return True if self is other else NotImplemented

    def __ne__(self, other):
        equal = self == other
        return not equal if equal is not NotImplemented else NotImplemented

    def __hash__(self):
        return pylir.intr.object.hash(self)

    def __repr__(self):
        # TODO: Use hex formatting when turning the id to a string
        name = pylir.intr.getSlot(type(self), type, "__name__")
        return "<" + name + " object at " + str(id(self)) + ">"

    def __str__(self):
        return repr(self)


@pylir.intr.const_export
class BaseException:
    __slots__ = pylir.intr.BaseException.__slots__

    def __new__(cls, *args, **kwargs):
        obj = pylir.intr.makeObject(cls)
        pylir.intr.setSlot(obj, BaseException, "args", args)
        return obj

    def __init__(self, *args):
        pylir.intr.setSlot(self, BaseException, "args", args)

    def __str__(self):
        args = pylir.intr.getSlot(self, BaseException, "args")
        if len(args) == 0:
            return ""
        if len(args) == 1:
            return str(args[0])
        return str(args)


@pylir.intr.const_export
class Exception(BaseException):
    pass


@pylir.intr.const_export
class TypeError(Exception):
    pass


@pylir.intr.const_export
class LookupError(Exception):
    pass


@pylir.intr.const_export
class IndexError(LookupError):
    pass


@pylir.intr.const_export
class KeyError(LookupError):
    pass


@pylir.intr.const_export
class ValueError(Exception):
    pass


@pylir.intr.const_export
class UnicodeError(ValueError):
    pass


@pylir.intr.const_export
class AssertionError(Exception):
    pass


@pylir.intr.const_export
class AttributeError(Exception):
    pass


@pylir.intr.const_export
class NotImplementedError(Exception):
    pass


@pylir.intr.const_export
class NameError(Exception):
    pass


@pylir.intr.const_export
class UnboundLocalError(NameError):
    pass


@pylir.intr.const_export
class ArithmeticError(Exception):
    pass


@pylir.intr.const_export
class OverflowError(ArithmeticError):
    pass


@pylir.intr.const_export
class StopIteration(Exception):
    __slots__ = "value"

    def __init__(self, *args):
        pylir.intr.setSlot(self, StopIteration, "args", args)
        if len(args) > 0:
            pylir.intr.setSlot(self, StopIteration, "value", args[0])
        else:
            pylir.intr.setSlot(self, StopIteration, "value", None)


@pylir.intr.const_export
class NoneType:
    def __new__(cls, *args, **kwargs):
        return None

    def __repr__(self):
        return "None"

    def __bool__(self):
        return False


@pylir.intr.const_export
class NotImplementedType:
    def __new__(cls, *args, **kwargs):
        return NotImplemented

    def __repr__(self):
        return "NotImplemented"


@pylir.intr.const_export
class function:
    __slots__ = pylir.intr.function.__slots__

    # The '/' making 'self' a positional parameter here is actually required
    # as otherwise we'd have a stack overflow! Calling __call__ would
    # do dictionary lookups for 'self' which lead to calls to __eq__ and
    # __hash__ functions which then lead back to here.
    def __call__(self, /, *args, **kwargs):
        return pylir.intr.function.call(self, self, args, kwargs)


@pylir.intr.const_export
class cell:
    __slots__ = "cell_contents"

    def __new__(cls, *args, **kwargs):
        obj = pylir.intr.makeObject(cls)
        # TODO: error if args is not equal to 0 or 1
        if len(args) == 1:
            pylir.intr.setSlot(obj, cell, "cell_contents", args[0])
        return obj


@pylir.intr.const_export
class dict:
    def __getitem__(self, item):
        # This awkward dance of using the walrus operator is required here
        # as storing the result and then using it as the argument of a function
        # would not work in the case that 'tryGetItem' returns an unbound value.
        # This way we can pipe the result into both 'res' and the intrinsic
        success = pylir.intr.isUnboundValue(
            res := pylir.intr.dict.tryGetItem(self, item, hash(item)))
        if not success:
            raise KeyError
        return res

    def __setitem__(self, key, value):
        pylir.intr.dict.setItem(self, key, hash(key), value)

    def __len__(self):
        return pylir.intr.dict.len(self)


@pylir.intr.const_export
class tuple:
    def __len__(self):
        return pylir.intr.tuple.len(self)

    def __getitem__(self, item):
        # TODO: negative indices, use index etc.
        return pylir.intr.tuple.getItem(self, item)

    def __repr__(self):
        if len(self) == 0:
            return "()"
        res = "(" + repr(self[0])
        if len(self) == 1:
            return res + ",)"
        i = 1
        while i < len(self):
            res += ", " + repr(self[i])
            i += 1
        return res + ")"


@pylir.intr.const_export
class int:
    def __repr__(self):
        return pylir.intr.int.toStr(self)

    # TODO: These are the bare minimum and do not take into account the type
    #       of other.
    def __add__(self, other):
        return pylir.intr.int.add(self, other)

    def __eq__(self, other):
        return pylir.intr.int.cmp("eq", self, other)

    def __ne__(self, other):
        return pylir.intr.int.cmp("ne", self, other)

    def __lt__(self, other):
        return pylir.intr.int.cmp("lt", self, other)

    def __le__(self, other):
        return pylir.intr.int.cmp("le", self, other)

    def __gt__(self, other):
        return pylir.intr.int.cmp("gt", self, other)

    def __ge__(self, other):
        return pylir.intr.int.cmp("ge", self, other)

    def __index__(self):
        return self

    def __bool__(self):
        return self != 0


@pylir.intr.const_export
class bool(int):

    def __new__(cls, arg=False):
        mro = pylir.intr.type.mro(type(arg))
        t = pylir.intr.mroLookup(mro, "__bool__")
        if not t[1]:
            mro = pylir.intr.type.mro(type(arg))
            t = pylir.intr.mroLookup(mro, "__len__")
            if not t[1]:
                return True
            res = unary_method_call(t[0], arg)
            mro = pylir.intr.type.mro(type(res))
            t = pylir.intr.mroLookup(mro, "__index__")
            if not t[1]:
                raise TypeError
            return unary_method_call(t[0], res) != 0
        return unary_method_call(t[0], arg)

    def __bool__(self):
        return self

    def __repr__(self):
        return "True" if self else "False"


@pylir.intr.const_export
class list:

    def __len__(self):
        return pylir.intr.list.len(self)

    def __getitem__(self, item):
        # TODO: negative indices, use index etc.
        return pylir.intr.list.getItem(self, item)


@pylir.intr.const_export
class str:

    def __new__(cls, object=None, encoding=None, errors=None):
        """
        Formal overloads to be detected:
        - str(object='')
        - str(object=b'', encoding='utf-8', errors='strict')
            if one of encoding or errors is specified
        """
        if encoding is None and errors is None:
            object = '' if object is None else object
            mro = pylir.intr.type.mro(type(object))
            t = pylir.intr.mroLookup(mro, "__str__")
            res = unary_method_call(t[0], object)
            if not isinstance(res, str):
                raise TypeError
            return res
        raise NotImplementedError

    # Positional only arguments are required for here and down below as the call
    # would otherwise do dictionary lookups for the keyword arguments, leading
    # to infinite recursion.
    def __hash__(self, /):
        return pylir.intr.str.hash(self)

    def __eq__(self, other, /):
        return pylir.intr.str.equal(self, other)

    def __str__(self, /):
        return self

    def __add__(self, other):
        return pylir.intr.str.concat(self, other)


@pylir.intr.const_export
def repr(arg):
    mro = pylir.intr.type.mro(type(arg))
    t = pylir.intr.mroLookup(mro, "__repr__")
    if not t[1]:
        raise TypeError
    res = unary_method_call(t[0], arg)
    if not isinstance(res, str):
        raise TypeError
    return res


@pylir.intr.const_export
def len(arg):
    mro = pylir.intr.type.mro(type(arg))
    t = pylir.intr.mroLookup(mro, "__len__")
    if not t[1]:
        raise TypeError
    res = unary_method_call(t[0], arg)
    mro = pylir.intr.type.mro(type(res))
    t = pylir.intr.mroLookup(mro, "__index__")
    if not t[1]:
        raise TypeError
    return unary_method_call(t[0], res)


@pylir.intr.const_export
def id(obj):
    return pylir.intr.object.id(obj)


@pylir.intr.const_export
def print(*objects, sep=None, end=None):
    # TODO: check sep & end are actually str if not None
    sep = " " if sep is None else sep
    end = "\n" if end is None else end
    i = 0
    tuple_len = len(objects)
    res = ""
    while i < tuple_len:
        if i != 0:
            res += sep
        res += str(objects[i])
        i += 1
    pylir.intr.intr.print(res + end)


@pylir.intr.const_export
def hash(obj, /):
    t = pylir.intr.mroLookup(pylir.intr.type.mro(type(obj)), "__hash__")
    if not t[1] or t[0] is None:
        raise TypeError
    # TODO: Check in range of sys.maxsize
    return unary_method_call(t[0], obj)


def object_isinstance(inst, cls, /):
    if pylir.intr.tuple.contains(pylir.intr.type.mro(type(cls)), type):
        return pylir.intr.tuple.contains(pylir.intr.type.mro(type(inst)), cls)
    # TODO: abstract
    raise NotImplementedError


@pylir.intr.const_export
def isinstance(inst, cls, /):
    if type(inst) is cls:
        return True

    cls_type = type(cls)
    if cls_type is type:
        return object_isinstance(inst, cls)

    if cls_type is tuple:
        # If the below causes stack overflow replace with intrinsics! Should be
        # fine as long as all methods called here don't use the tuple form of
        # isinstance.
        for it in cls:
            if isinstance(inst, it):
                return True
        return False

    t = pylir.intr.mroLookup(pylir.intr.type.mro(cls_type), "__instancecheck__")
    if t[1]:
        return binary_method_call(t[0], cls, inst)

    return object_isinstance(inst, cls)


@pylir.intr.const_export
class SeqIter:
    __slots__ = ("__seq", "__i")

    def __init__(self, seq) -> None:
        pylir.intr.setSlot(self, SeqIter, "__seq", seq)
        pylir.intr.setSlot(self, SeqIter, "__i", 0)

    def __iter__(self):
        return self

    def __next__(self):
        index = pylir.intr.getSlot(self, SeqIter, "__i")
        seq = pylir.intr.getSlot(self, SeqIter, "__seq")
        if index >= len(seq):
            raise StopIteration
        item = seq[index]
        pylir.intr.setSlot(self, SeqIter, "__i", index + 1)
        return item


@pylir.intr.const_export
def iter(*args):
    if not (1 <= len(args) <= 2):
        raise TypeError

    if len(args) == 1:
        obj = args[0]
        mro = pylir.intr.type.mro(type(obj))
        t = pylir.intr.mroLookup(mro, "__iter__")
        if t[1]:
            if t[0] is None:
                raise TypeError
            return unary_method_call(t[0], obj)
        t = pylir.intr.mroLookup(mro, "__getitem__")
        if not t[1]:
            raise TypeError
        return SeqIter(obj)

    raise NotImplementedError


@pylir.intr.const_export
def next(*args):
    if not (1 <= len(args) <= 2):
        raise TypeError
    obj = args[0]
    mro = pylir.intr.type.mro(type(obj))
    t = pylir.intr.mroLookup(mro, "__next__")
    if not t[1]:
        raise TypeError

    try:
        return unary_method_call(t[0], obj)
    except StopIteration as e:
        if len(args) == 2:
            return args[1]
        raise e
