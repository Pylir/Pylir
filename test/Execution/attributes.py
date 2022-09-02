#  // Copyright 2022 Markus Böck
#  //
#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: pylir %s -o %t -O3
# RUN: %t | FileCheck %s --match-full-lines

obj = object()

try:
    print(obj.thing)
    print("Failure")
except AttributeError:
    print("Success")

# CHECK: Success

try:
    obj.thing = 3
    print("Failure")
except AttributeError:
    print("Success")

# CHECK: Success
