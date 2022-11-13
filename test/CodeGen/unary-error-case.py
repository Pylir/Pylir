# RUN:  pylir %s -emit-pylir -o - -S | FileCheck %s

if not a:
    pass

# CHECK-LABEL: func @__init__()
# CHECK-NOT: ^{{[[:alnum:]]+}}
# CHECK: py.raise