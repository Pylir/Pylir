# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def test(iterable):
    a, b = iterable
    *a, b = a
    a, *b, c = b
    a, *b = c
    return a, b

# CHECK-LABEL: py.func private @"test$impl[0]"
# CHECK-SAME: %{{[[:alnum:]]+}}
# CHECK-SAME: %[[ITERABLE:[[:alnum:]]+]]
# CHECK: %[[AB:[[:alnum:]]+]]:2 = unpack %[[ITERABLE]] : (!py.dynamic, !py.dynamic)
# CHECK: %[[A:.*]], %[[B:.*]] = unpack %[[AB]]#0 : (), !py.dynamic, (!py.dynamic)
# CHECK: %[[A:.*]], %[[B_2:.*]], %[[C:.*]] = unpack %[[B]] : (!py.dynamic), !py.dynamic, (!py.dynamic)
# CHECK: %[[A:.*]], %[[B:.*]] = unpack %[[C]] : (!py.dynamic), !py.dynamic
# CHECK: %[[TUPLE:.*]] = makeTuple (%[[A]], %[[B]])
# CHECK: return %[[TUPLE]]
