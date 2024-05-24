#  Licensed under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

def const_export(obj) -> type:
    pass

def typeOf(obj: object) -> type:
    pass

def makeObject(cls: type) -> type:
    pass

def getSlot(obj: object, slot: int):
    pass

def setSlot(obj: object, slot: int, value: object):
    pass

def isUnboundValue(obj: object) -> bool:
    pass

def mroLookup(mro_tuple, slot: int):
    pass


def makeType(name: str, bases: tuple, slots: tuple):
    pass
