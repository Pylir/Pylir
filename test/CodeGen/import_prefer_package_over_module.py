# RUN: split-file %s %t
# RUN: pylir %t/main.py -S -emit-pylir -o - | FileCheck %s
# RUN: pylir %t/main.py -S -emit-pylir -o - -Xsingle-threaded | FileCheck %s

#--- main.py

import foo

# CHECK: call @foo.__init__

#--- foo/__init__.py

print("Package")

#--- foo.py

print("Module")

# CHECK: py.func @foo.__init__
# CHECK: constant(#py.str<"Package">)
