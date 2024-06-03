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
    if not pylir.intr.isUnboundValue(
        t := pylir.intr.mroLookup(mro, pylir.intr.type.__get__)
    ):
        method = t(obj, type(obj))
    # TODO: This is incorrect. One should not be passing any arguments, and obj
    #       should be bound as the self parameter by the __get__ implementation
    #       of function. Until this is implemented we are doing this
    #       incorrectly.
    return method(obj)


def binary_method_call(method, self, other):
    mro = pylir.intr.type.mro(type(method))
    if not pylir.intr.isUnboundValue(
        t := pylir.intr.mroLookup(mro, pylir.intr.type.__get__)
    ):
        method = t(self, type(self))
    # TODO: This is incorrect; See unary_method call description for details.
    #       Should be `method(other)` in the future.
    return method(self, other)


@pylir.intr.const_export
class type:
    __slots__ = pylir.intr.type.__slots__

    def __new__(cls, name, bases, dict, **kwargs):
        # TODO: Assign instance slots from 'dict', create instance of metatype
        #  not 'type', add '__weakref__' slot by default, allow 0 len 'bases',
        #  call '__set_name__' of descriptors and '__init_subclass__' of
        #  'super()'.

        try:
            # TODO: Verify slots are all strings.
            slots = (*dict['__slots__'],)
        except KeyError:
            slots = ('__dict__',)

        result = pylir.intr.makeType(name, bases, slots)
        pylir.intr.setSlot(result, pylir.intr.type.__dict__, dict)
        return result

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
        new_method = pylir.intr.mroLookup(mro, pylir.intr.type.__new__)
        # TODO: This is not the proper sequence to call a function and only
        #       works for `function`, but not arbitrary callables.
        #       Proper would be 'new_method(new_method,*args,**kwargs)' once
        #       unpacking is supported.
        res = pylir.intr.function.call(
            new_method, new_method, pylir.intr.tuple.prepend(self, args), kwargs
        )
        res_type = pylir.intr.typeOf(res)
        mro = pylir.intr.type.mro(res_type)
        if not pylir.intr.tuple.contains(mro, self):
            return res
        init_method = pylir.intr.mroLookup(mro, pylir.intr.type.__init__)
        # TODO: Same as above but:
        #       'init_method(init_method, res, *args, **kwargs)'
        init_ret = pylir.intr.function.call(
            init_method, init_method, pylir.intr.tuple.prepend(res, args), kwargs
        )
        if init_ret is not None:
            raise TypeError
        return res


@pylir.intr.const_export
class object:
    def __new__(cls, *args, **kwargs):
        result = pylir.intr.makeObject(cls)
        slots = pylir.intr.type.slots(cls)
        # Note: Be very careful that the below code cannot possibly call
        # 'object.__new__'.
        i = 0
        while i < len(slots):
            if slots[i] == '__dict__':
                pylir.intr.setSlot(result, i, {})
                break

            i += 1

        return result

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
        name = pylir.intr.getSlot(type(self), pylir.intr.type.__name__)
        return "<" + name + " object at " + str(id(self)) + ">"

    def __str__(self):
        return repr(self)

    def __getattribute__(self, item):
        if not isinstance(item, str):
            raise TypeError

        self_type = type(self)
        mro = pylir.intr.type.mro(self_type)

        NOT_FOUND_SENTINEL = object()

        type_lookup = NOT_FOUND_SENTINEL
        for i in mro:
            # TODO: __dict__ likely can't be unbound in the future.
            if not pylir.intr.isUnboundValue(
                d := pylir.intr.getSlot(i, pylir.intr.type.__dict__)
            ):
                if not pylir.intr.isUnboundValue(
                    attr_lookup := pylir.intr.dict.tryGetItem(d, item, hash(item))
                ):
                    type_lookup = attr_lookup
                    break

        # Descriptor that has both __get__ and __set__ takes priority over
        # __dict__. If it has just __get__ it will be called below if the
        # object does not have a __dict__.
        if type_lookup is not NOT_FOUND_SENTINEL:
            type_lookup_mro = pylir.intr.type.mro(type(type_lookup))
            if not pylir.intr.isUnboundValue(
                getter := pylir.intr.mroLookup(type_lookup_mro, pylir.intr.type.__get__)
            ) and not pylir.intr.isUnboundValue(
                pylir.intr.mroLookup(type_lookup_mro, pylir.intr.type.__set__)
            ):
                return getter(self, self_type)

        slots = pylir.intr.type.slots(self_type)
        i = 0
        # Find the index of the __dict__ slot if it exists
        while i < len(slots):
            if slots[i] != "__dict__":
                i += 1
                continue

            # TODO: __dict__ likely can't be unbound in the future.
            if not pylir.intr.isUnboundValue(d := pylir.intr.getSlot(self, i)):
                try:
                    return d[item]
                except KeyError:
                    pass
            break

        if type_lookup is not NOT_FOUND_SENTINEL:
            if not pylir.intr.isUnboundValue(
                getter := pylir.intr.mroLookup(mro, pylir.intr.type.__get__)
            ):
                return getter(self, self_type)

            return type_lookup

        raise AttributeError

    def __getattr__(self, item):
        # __getattr__ is called as a fallback when __getattribute__ fails with
        # a AttributeError. The default fallback doesn't do anything but fail as
        # well.
        raise AttributeError

    def __setattr__(self, key, value):
        if not isinstance(key, str):
            raise TypeError

        self_type = type(self)
        mro = pylir.intr.type.mro(self_type)

        NOT_FOUND_SENTINEL = object()

        type_lookup = NOT_FOUND_SENTINEL
        for i in mro:
            # TODO: __dict__ likely can't be unbound in the future.
            if not pylir.intr.isUnboundValue(
                d := pylir.intr.getSlot(i, pylir.intr.type.__dict__)
            ):
                if not pylir.intr.isUnboundValue(
                    attr_lookup := pylir.intr.dict.tryGetItem(d, key, hash(key))
                ):
                    type_lookup = attr_lookup
                    break

        if type_lookup is not NOT_FOUND_SENTINEL:
            type_lookup_mro = pylir.intr.type.mro(type(type_lookup))
            if not pylir.intr.isUnboundValue(
                setter := pylir.intr.mroLookup(type_lookup_mro, pylir.intr.type.__set__)
            ):
                return setter(self, value)

        slots = pylir.intr.type.slots(self_type)
        i = 0
        # Find the index of the __dict__ slot if it exists
        while i < len(slots):
            if slots[i] != "__dict__":
                i += 1
                continue

            # TODO: __dict__ likely can't be unbound in the future.
            if not pylir.intr.isUnboundValue(d := pylir.intr.getSlot(self, i)):
                d[key] = value
                return
            i += 1

        raise AttributeError


@pylir.intr.const_export
class BaseException:
    __slots__ = pylir.intr.BaseException.__slots__

    def __new__(cls, *args, **kwargs):
        obj = pylir.intr.makeObject(cls)
        pylir.intr.setSlot(obj, pylir.intr.BaseException.args, args)
        return obj

    def __init__(self, *args):
        pylir.intr.setSlot(self, pylir.intr.BaseException.args, args)

    def __str__(self):
        args = pylir.intr.getSlot(self, pylir.intr.BaseException.args)
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
        pylir.intr.setSlot(self, pylir.intr.BaseException.args, args)
        if len(args) > 0:
            pylir.intr.setSlot(self, len(pylir.intr.BaseException.__slots__), args[0])
        else:
            pylir.intr.setSlot(self, len(pylir.intr.BaseException.__slots__), None)


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
class module:
    __slots__ = "__dict__"

@pylir.intr.const_export
class cell:
    __slots__ = "cell_contents"

    def __new__(cls, *args, **kwargs):
        obj = pylir.intr.makeObject(cls)
        # TODO: error if args is not equal to 0 or 1
        if len(args) == 1:
            pylir.intr.setSlot(obj, 0, args[0])
        return obj


@pylir.intr.const_export
class dict:
    def __getitem__(self, item):
        # This awkward dance of using the walrus operator is required here
        # as storing the result and then using it as the argument of a function
        # would not work in the case that 'tryGetItem' returns an unbound value.
        # This way we can pipe the result into both 'res' and the intrinsic
        failed = pylir.intr.isUnboundValue(
            res := pylir.intr.dict.tryGetItem(self, item, hash(item))
        )
        if failed:
            raise KeyError
        return res

    def __contains__(self, item):
        return not pylir.intr.isUnboundValue(
            pylir.intr.dict.tryGetItem(self, item, hash(item))
        )

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
        if pylir.intr.isUnboundValue(
            t := pylir.intr.mroLookup(mro, pylir.intr.type.__bool__)
        ):
            mro = pylir.intr.type.mro(type(arg))
            if pylir.intr.isUnboundValue(
                t := pylir.intr.mroLookup(mro, pylir.intr.type.__len__)
            ):
                return True
            res = unary_method_call(t, arg)
            mro = pylir.intr.type.mro(type(res))
            if pylir.intr.isUnboundValue(
                t := pylir.intr.mroLookup(mro, pylir.intr.type.__index__)
            ):
                raise TypeError
            return unary_method_call(t, res) != 0
        return unary_method_call(t, arg)

    def __bool__(self):
        return self

    def __repr__(self):
        return "True" if self else "False"


@pylir.intr.const_export
class float:
    pass


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
            object = "" if object is None else object
            mro = pylir.intr.type.mro(type(object))
            t = pylir.intr.mroLookup(mro, pylir.intr.type.__str__)
            res = unary_method_call(t, object)
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
    if pylir.intr.isUnboundValue(
        t := pylir.intr.mroLookup(mro, pylir.intr.type.__repr__)
    ):
        raise TypeError
    res = unary_method_call(t, arg)
    if not isinstance(res, str):
        raise TypeError
    return res


@pylir.intr.const_export
def len(arg):
    mro = pylir.intr.type.mro(type(arg))
    if pylir.intr.isUnboundValue(
        t := pylir.intr.mroLookup(mro, pylir.intr.type.__len__)
    ):
        raise TypeError
    res = unary_method_call(t, arg)
    mro = pylir.intr.type.mro(type(res))
    if pylir.intr.isUnboundValue(
        t := pylir.intr.mroLookup(mro, pylir.intr.type.__index__)
    ):
        raise TypeError
    return unary_method_call(t, res)


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
    mro = pylir.intr.type.mro(type(obj))
    if (
        pylir.intr.isUnboundValue(
            t := pylir.intr.mroLookup(mro, pylir.intr.type.__hash__)
        )
        or t is None
    ):
        raise TypeError
    # TODO: Check in range of sys.maxsize
    return unary_method_call(t, obj)


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

    mro = pylir.intr.type.mro(cls_type)
    if not pylir.intr.isUnboundValue(
        t := pylir.intr.mroLookup(mro, pylir.intr.type.__instancecheck__)
    ):
        return binary_method_call(t, cls, inst)

    return object_isinstance(inst, cls)


@pylir.intr.const_export
class SeqIter:
    __slots__ = ("__seq", "__i")

    def __init__(self, seq) -> None:
        pylir.intr.setSlot(self, 0, seq)
        pylir.intr.setSlot(self, 1, 0)

    def __iter__(self):
        return self

    def __next__(self):
        index = pylir.intr.getSlot(self, 1)
        seq = pylir.intr.getSlot(self, 0)
        if index >= len(seq):
            raise StopIteration
        item = seq[index]
        pylir.intr.setSlot(self, 1, index + 1)
        return item


@pylir.intr.const_export
def iter(*args):
    if not (1 <= len(args) <= 2):
        raise TypeError

    if len(args) == 1:
        obj = args[0]
        mro = pylir.intr.type.mro(type(obj))
        if not pylir.intr.isUnboundValue(
            t := pylir.intr.mroLookup(mro, pylir.intr.type.__iter__)
        ):
            if t is None:
                raise TypeError
            return unary_method_call(t, obj)

        if pylir.intr.isUnboundValue(
            pylir.intr.mroLookup(mro, pylir.intr.type.__getitem__)
        ):
            raise TypeError
        return SeqIter(obj)

    raise NotImplementedError


@pylir.intr.const_export
def next(*args):
    if not (1 <= len(args) <= 2):
        raise TypeError
    obj = args[0]
    mro = pylir.intr.type.mro(type(obj))
    if pylir.intr.isUnboundValue(
        t := pylir.intr.mroLookup(mro, pylir.intr.type.__next__)
    ):
        raise TypeError

    try:
        return unary_method_call(t, obj)
    except StopIteration as e:
        if len(args) == 2:
            return args[1]
        raise e


@pylir.intr.const_export
def __build_class__(func, name, /, *bases, metaclass=type, **kwds):
    # TODO: Compute MRO order, compute metatype,

    # TODO: Initialize dictionary with __prepare__.
    d = {}
    func(d)
    bases = *bases, object
    return metaclass(name, bases, d)
