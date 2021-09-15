// RUN: pylir-opt %s --verify-diagnostics

func @test1() -> !py.dynamic {
    %0 = py.constant #py.tuple<()>
    %1 = py.constant #py.dict<{}>
    %2 = py.constant #py.int<0>
    %3 = py.call %2( * %0, * * %1)
    return %3 : !py.dynamic
}

func @test2() -> !py.dynamic {
    %0 = py.makeTuple()
    %1 = py.makeDict()
    %2 = py.constant #py.int<0>
    %3 = py.call %2(*%0, **%1)
    return %3 : !py.dynamic
}

func @test3() -> !py.dynamic {
    %0 = py.constant #py.int<0>
    %1 = py.makeDict()
    %2 = py.constant #py.int<0>
    %3 = py.call %2(*%0, **%1) //expected-error {{tuple and dict operands must be makeTuple, makeDict or tuple or dict constant ops}}
    return %3 : !py.dynamic
}
