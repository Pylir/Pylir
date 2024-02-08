#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: pylir %s -Xnew-codegen -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$BASE_EXCEPTION:.*]] = #py.globalValue<builtins.BaseException{{(,|>)}}

# CHECK: %[[GLOBALS:.*]] = py.makeDict

# CHECK: %[[STR:.*]] = py.constant(#py.str<"BaseException">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: %[[ITEM:.*]] = py.dict_tryGetItem %[[GLOBALS]][%[[STR]] hash(%[[HASH]])]
# CHECK: %[[BUILTIN:.*]] = py.constant(#[[$BASE_EXCEPTION]])
# CHECK: %[[IS_UNBOUND:.*]] = py.isUnboundValue %[[ITEM]]
# CHECK: %[[SELECT:.*]] = arith.select %[[IS_UNBOUND:.*]], %[[BUILTIN]], %[[ITEM]]
# CHECK: func "__main__.foo"(%{{.*}} "x" = %[[SELECT]])
def foo(x=BaseException):
    # CHECK: %[[FIVE:.*]] = py.constant(#py.int<5>)
    # CHECK: return %[[FIVE]]
    TypeError = 5
    return TypeError
