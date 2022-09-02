# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: func.func private @"test$impl[0]"
def test():
    test.thing = 5
    # CHECK: %[[FIVE:.*]] = py.constant(#py.int<5>)
    # CHECK: %[[LOAD:.*]] = py.load @test$handle
    # CHECK: %[[C:.*]] = py.constant(#py.str<"thing">)
    # CHECK: py.call @pylir__setattr__(%[[LOAD]], %[[C]], %[[FIVE]])

    return test.thing
    # CHECK: %[[LOAD:.*]] = py.load @test$handle
    # CHECK: %[[C:.*]] = py.constant(#py.str<"thing">)
    # CHECK: %[[RES:.*]] = py.call @pylir__getattribute__(%[[LOAD]], %[[C]])
    # CHECK: return %[[RES]]
