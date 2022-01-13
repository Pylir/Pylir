// RUN: pylir-opt %s --test-memory-ssa --split-input-file

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func @test() -> index {
    %0 = py.constant #py.str<"test">
    %1 = py.makeList ()
    py.list.append %1, %0
    %2 = py.list.len %1
    return %2 : index
}
