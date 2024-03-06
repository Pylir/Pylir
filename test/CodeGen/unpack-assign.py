# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s


# CHECK-LABEL: func "__main__.test"
# CHECK-SAME: %[[ITERABLE:[[:alnum:]]+]]
def test(iterable):
    # CHECK: %[[AB:[[:alnum:]]+]]:2 = py.unpack %[[ITERABLE]] : (!py.dynamic, !py.dynamic)
    a, b = iterable
    # CHECK: %[[A:.*]], %[[B:.*]] = py.unpack %[[AB]]#0 : (), !py.dynamic, (!py.dynamic)
    *a, b = a
    # CHECK: %[[A:.*]], %[[B_2:.*]], %[[C:.*]] = py.unpack %[[B]] : (!py.dynamic), !py.dynamic, (!py.dynamic)
    a, *b, c = b
    # CHECK: %[[A:.*]], %[[B:.*]] = py.unpack %[[C]] : (!py.dynamic), !py.dynamic
    a, *b = c
