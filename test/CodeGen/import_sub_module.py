# RUN: split-file %s %t
# RUN: pylir %t/main.py -I %t/wrong -I %t/right -S -emit-pylir -o - | FileCheck %s

#--- main.py

import foo.bar.baz

# CHECK: call @foo.__init__
# CHECK: call @foo.bar.__init__
# CHECK: call @foo.bar.baz.__init__

#--- right/foo/__init__.py

#--- right/foo/bar/__init__.py

#--- right/foo/bar/baz.py

print(True)

# CHECK: py.func @foo.bar.baz.__init__()
# CHECK: constant(#py.bool<True>)

#--- wrong/foo/bar/baz.py

print(False)
