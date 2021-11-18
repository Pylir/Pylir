# RUN: pylir %s -emit-mlir -o - | FileCheck %s

5
# CHECK: py.constant #py.int<5>

6.5
# CHECK: py.constant 6.5

'text'
# CHECK: py.constant "text"

True

# CHECK: py.constant #py.bool<True>

False

# CHECK: py.constant #py.bool<False>

None
# CHECK: py.constant @builtins.None
