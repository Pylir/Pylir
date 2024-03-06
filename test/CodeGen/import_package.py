#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: split-file %s %t
# RUN: pylir %t/main.py -S -emit-pylir -o - | FileCheck %s

#--- main.py

import foo

# CHECK: initModule @foo

#--- foo/__init__.py
