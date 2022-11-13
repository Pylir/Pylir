# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def foo(slots, item):
    i = 0
    while True:
        if slots[i] != "__dict__":
            i += 1
            continue

        try:
            return item + 0
        except:
            pass

# CHECK-LABEL: func private @"foo$impl[0]"
# CHECK: %[[ZERO:.*]] = py.constant(#py.int<0>)
# CHECK-NEXT: cf.br ^[[BODY:.*]](%[[ZERO]] : !py.dynamic)
# CHECK-NEXT: ^[[BODY]](%{{.*}}: !py.dynamic loc({{.*}})):