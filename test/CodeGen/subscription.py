# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

# CHECK-LABEL: func "__main__.test"
# CHECK-SAME: %[[A:[[:alnum:]]+]]
# CHECK-SAME: %[[B:[[:alnum:]]+]]
# CHECK-SAME: %[[C:[[:alnum:]]+]]
def test(a, b, c):
    # CHECK: %[[A_C:.*]] = getItem %[[A]][%[[C]]]
    # CHECK: setItem %[[A]][%[[B]]] to %[[A_C]]
    a[b] = a[c]
