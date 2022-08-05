#  // Copyright 2022 Markus Böck
#  //
#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pylir.intr.BaseException
import pylir.intr.object
import pylir.intr.str
import pylir.intr.type
import pylir.intr.function


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
        return pylir.intr.str.concat("<", name, " object at ", str(id(self)),
                                     ">")

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
    __slots__ = ("value",)

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
    # as otherwise we'd have a stack overflow! Calling __call__ would otherwise
    # do dictionary lookups for 'self' which lead to calls to __eq__ and
    # __hash__ of functions which then lead back to here.
    def __call__(self, /, *args, **kwargs):
        return pylir.intr.function.call(self, self, args, kwargs)


@pylir.intr.const_export
def id(obj):
    return pylir.intr.object.id(obj)
