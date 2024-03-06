# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

# CHECK-LABEL: func "__main__.test"
# CHECK-SAME: %[[A:[[:alnum:]]+]]
def test(a):
    # CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
    # CHECK: setAttr "test" of %[[A]] to %[[THREE]]
    a.test = 3
    # CHECK: %[[RES:.*]] = getAttribute "test" of %[[A]]
    # CHECK: return %[[RES]]
    return a.test
