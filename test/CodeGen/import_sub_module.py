#  // Licensed under the Apache License v2.0 with LLVM Exceptions.
#  // See https://llvm.org/LICENSE.txt for license information.
#  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: split-file %s %t
# RUN: pylir %t/main.py -I %t/wrong -I %t/right -S -emit-pylir -o - | FileCheck %s

#--- main.py

import foo.bar.baz

# CHECK: initModule @foo
# CHECK: initModule @foo.bar
# CHECK: initModule @foo.bar.baz

#--- right/foo/__init__.py

#--- right/foo/bar/__init__.py

#--- right/foo/bar/baz.py

print(True)

# CHECK: init "foo.bar.baz"
# CHECK: constant(#py.bool<True>)

#--- wrong/foo/bar/baz.py

print(False)
