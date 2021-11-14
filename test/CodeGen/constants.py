# RUN: pylir %s -emit-mlir -o - | FileCheck %s

5
# CHECK: constant #py.int<5>

6.5
# CHECK: constant 6.5

'text'
# CHECK: constant "text"

True

# CHECK: constant #py.bool<True>

False

# CHECK: constant #py.bool<False>

None
# CHECK: py.constant @builtins.None
