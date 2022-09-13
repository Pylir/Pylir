#  // Copyright 2022 Markus Böck
#  //
#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: __init__

# CHECK: %[[RES:.*]] = py.makeFunc @"foo$cc[0]"
# CHECK: %[[NAME:.*]] = py.constant(#py.str<"foo">)
# CHECK: py.setSlot "__qualname__" of %[[RES]] : %{{.*}} to %[[NAME]]
# CHECK: %[[DEFAULTS:.*]] = py.constant(@builtins.None)
# CHECK: py.setSlot "__defaults__" of %[[RES]] : %{{.*}} to %[[DEFAULTS]]
# CHECK: %[[KWDEFAULTS:.*]] = py.constant(@builtins.None)
# CHECK: py.setSlot "__kwdefaults__" of %[[RES]] : %{{.*}} to %[[KWDEFAULTS]]
# CHECK: %[[CLOSURE:.*]] = py.constant(@builtins.None)
# CHECK: py.setSlot "__closure__" of %[[RES]] : %{{.*}} to %[[CLOSURE]]
# CHECK: py.store %[[RES]] into @foo

def foo():
    y = 5
    x = 3

    lambda a=3, *, c=1: x + y

# CHECK-LABEL: func private @"foo$impl[0]"
# CHECK: %[[CELL_TYPE:.*]] = py.constant(@builtins.cell)
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[CELL_TYPE]])
# CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
# CHECK: %[[NEW:.*]] = py.getSlot "__new__" from %[[CELL_TYPE]]
# CHECK: %[[Y:.*]] = py.function.call %[[NEW]](%[[NEW]], %[[TUPLE]], %[[DICT]])
# CHECK: %[[CELL_TYPE:.*]] = py.constant(@builtins.cell)
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[CELL_TYPE]])
# CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
# CHECK: %[[NEW:.*]] = py.getSlot "__new__" from %[[CELL_TYPE]]
# CHECK: %[[X:.*]] = py.function.call %[[NEW]](%[[NEW]], %[[TUPLE]], %[[DICT]])
# CHECK: %[[FIVE:.*]] = py.constant(#py.int<5>)
# CHECK: py.setSlot "cell_contents" of %[[Y]] : %{{.*}} to %[[FIVE]]
# CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK: py.setSlot "cell_contents" of %[[X]] : %{{.*}} to %[[THREE]]
# CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK: %[[ONE:.*]] = py.constant(#py.int<1>)
# CHECK: %[[C:.*]] = py.constant(#py.str<"c">)
# CHECK: %[[C_HASH:.*]] = py.str.hash %[[C]]
# CHECK: %[[RES:.*]] = py.makeFunc @"foo.<locals>.<lambda>$cc[0]"
# CHECK: %[[NAME:.*]] = py.constant(#py.str<"foo.<locals>.<lambda>">)
# CHECK: py.setSlot "__qualname__" of %[[RES]] : %{{.*}} to %[[NAME]]
# CHECK: %[[DEFAULTS:.*]] = py.makeTuple (%[[THREE]])
# CHECK: py.setSlot "__defaults__" of %[[RES]] : %{{.*}} to %[[DEFAULTS]]
# CHECK: %[[KWDEFAULTS:.*]] = py.makeDict (%[[C]] hash(%[[C_HASH]]) : %[[ONE]])
# CHECK: py.setSlot "__kwdefaults__" of %[[RES]] : %{{.*}} to %[[KWDEFAULTS]]
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[X]], %[[Y]])
# CHECK: py.setSlot "__closure__" of %[[RES]] : %{{.*}} to %[[TUPLE]]

# CHECK: func private @"foo.<locals>.<lambda>$impl[0]"
# CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
# CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
# CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
# CHECK: %[[CLOSURE:.*]] = py.getSlot "__closure__" from %[[ARG0]]
# CHECK: %[[C0:.*]] = arith.constant 0
# CHECK: %[[X_CELL:.*]] = py.tuple.getItem %[[CLOSURE]][%[[C0]]]
# CHECK: %[[C1:.*]] = arith.constant 1
# CHECK: %[[Y_CELL:.*]] = py.tuple.getItem %[[CLOSURE]][%[[C1]]]
# CHECK: %[[X:.*]] = py.getSlot "cell_contents" from %[[X_CELL]]
# CHECK: %[[X_UNBOUND:.*]] = py.isUnboundValue %[[X]]
# CHECK: cf.cond_br %[[X_UNBOUND]], ^[[RAISE_BLOCK:.*]], ^[[SUCCESS_BLOCK:[[:alnum:]]+]]
# CHECK: ^[[RAISE_BLOCK]]:
# CHECK: raise
# CHECK: ^[[SUCCESS_BLOCK]]:
# CHECK: %[[Y:.*]] = py.getSlot "cell_contents" from %[[Y_CELL]]
# CHECK: %[[RES:.*]] = py.call @pylir__add__(%[[X]], %[[Y]])
# CHECK: return %[[RES]]
