#  // Copyright 2022 Markus Böck
#  //
#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pylir.intr.object
import pylir.intr.type
import pylir.intr.str


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
        name = pylir.intr.getSlot(type(self), type, "__name__")
        return pylir.intr.str.concat("<", name, " object at ", str(id(self)),
                                     ">")

    def __str__(self):
        return repr(self)


@pylir.intr.const_export
def id(obj):
    return pylir.intr.object.id(obj)
