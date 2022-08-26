# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def test(iterable):
    a, b = iterable
    *a, b = a
    a, *b, c = b
    a, *b = c
    return a, b

# CHECK-LABEL: func.func private @"test$impl[0]"
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: %[[ITERABLE:[[:alnum:]]+]]
# CHECK: %[[AB:[[:alnum:]]+]]:2 = py.unpack %[[ITERABLE]] : (!py.dynamic, !py.dynamic)
# CHECK: %[[A:.*]], %[[B:.*]] = py.unpack %[[AB]]#0 : (), !py.dynamic, (!py.dynamic)
# CHECK: %[[A:.*]], %[[B_2:.*]], %[[C:.*]] = py.unpack %[[B]] : (!py.dynamic), !py.dynamic, (!py.dynamic)
# CHECK: %[[A:.*]], %[[B:.*]] = py.unpack %[[C]] : (!py.dynamic), !py.dynamic
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[A]], %[[B]])
# CHECK: return %[[TUPLE]]
