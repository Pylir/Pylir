# RUN: pylir %s -emit-mlir -o - | FileCheck %s

try:
    pass
except 0:
    pass

# CHECK-LABEL: __init__
# CHECK-NEXT: %[[VALUE:.*]] =
# CHECK-NEXT: return %[[VALUE]]
