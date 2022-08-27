#  // Copyright 2022 Markus Böck
#  //
#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: pylir %s -o %t -O3
# RUN: %t | FileCheck %s --match-full-lines


def outer(x):
    return lambda y: x + y


f = outer(3)
print(f(4))
# CHECK: 7

print(f(10))
# CHECK: 13
