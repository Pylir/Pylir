# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

5
# CHECK: py.constant #py.int<value = 5>

6.5
# CHECK: py.constant #py.float<value = 6.500000e+00>

'text'
# CHECK: py.constant #py.str<value = "text">

True

# CHECK: py.constant #py.bool<value = True>

False

# CHECK: py.constant #py.bool<value = False>

None
# CHECK: py.constant @builtins.None
