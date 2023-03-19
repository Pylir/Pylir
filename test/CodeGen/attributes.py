# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: py.func private @"test$impl[0]"
def test():
    test.thing = 5
    # CHECK: %[[FIVE:.*]] = constant(#py.int<5>)
    # CHECK: %[[LOAD:.*]] = load @test$handle
    # CHECK: %[[C:.*]] = constant(#py.str<"thing">)
    # CHECK: call @pylir__setattr__(%[[LOAD]], %[[C]], %[[FIVE]])

    return test.thing
    # CHECK: %[[LOAD:.*]] = load @test$handle
    # CHECK: %[[C:.*]] = constant(#py.str<"thing">)
    # CHECK: %[[RES:.*]] = call @pylir__getattribute__(%[[LOAD]], %[[C]])
    # CHECK: return %[[RES]]
