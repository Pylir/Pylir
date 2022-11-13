# RUN: pylir %s -emit-pylir -S -o - | FileCheck %s

print(a)

# CHECK-LABEL: func @__init__()
# CHECK-NOT: ^{{[[:alnum:]]+}}
# CHECK: py.raise