# RUN: split-file %s %t
# RUN: pylir %t/main.py -S -emit-pylir -o - | FileCheck %s

#--- main.py

import foo

# CHECK: call @foo.__init__

#--- foo/__init__.py
