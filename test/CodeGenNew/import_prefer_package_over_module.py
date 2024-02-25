#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: split-file %s %t
# RUN: pylir -Xnew-codegen %t/main.py -S -emit-pylir -o - | FileCheck %s
# RUN: pylir -Xnew-codegen %t/main.py -S -emit-pylir -o - -Xsingle-threaded | FileCheck %s

#--- main.py

import foo

# CHECK: initModule @foo

#--- foo/__init__.py

print("Package")

#--- foo.py

print("Module")

# CHECK: init "foo"
# CHECK: constant(#py.str<"Package">)
