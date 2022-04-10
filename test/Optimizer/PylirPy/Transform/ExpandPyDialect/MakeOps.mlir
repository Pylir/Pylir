// RUN: pylir-opt %s --pylir-expand-py-dialect --split-input-file | FileCheck %s

// XFAIL: *

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.TypeError =  #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.function =  #py.type
py.globalValue @builtins.StopIteration = #py.type

func @make_list_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant #py.int<3>
    %1 = py.constant #py.int<4>
    %2 = py.makeList (%0, *%arg0, %1)
    return %2 : !py.dynamic
}

// CHECK-LABEL: @make_list_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = py.constant #py.int<3>
// CHECK: %[[FOUR:.*]] = py.constant #py.int<4>
// CHECK: %[[LIST:.*]] = py.makeList (%[[THREE]])
// ... __iter__ call, __next__ call, exception handling of StopIteration. Basically same as ForStmt
// CHECK: %[[NEXT:.*]] = py.invoke_indirect %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-NOT: %{{[[:alnum:]]+}}
// CHECK-NEXT: label ^[[HAPPY_PATH:[[:alnum:]]+]]
// CHECK: ^[[HAPPY_PATH]]
// CHECK: py.list.append %[[LIST]], %[[NEXT]]
// ...
// CHECK: py.list.append %[[LIST]], %[[FOUR]]
// CHECK: return %[[LIST]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.TypeError =  #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.function =  #py.type
py.globalValue @builtins.StopIteration = #py.type

func @make_tuple_op(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant #py.int<3>
    %1 = py.constant #py.int<4>
    %2 = py.makeTuple (%0, *%arg0, %1)
    return %2 : !py.dynamic
}

// CHECK-LABEL: @make_tuple_op
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK: %[[THREE:.*]] = py.constant #py.int<3>
// CHECK: %[[FOUR:.*]] = py.constant #py.int<4>
// CHECK: %[[LIST:.*]] = py.makeList (%[[THREE]])
// ... __iter__ call, __next__ call, exception handling of StopIteration. Basically same as ForStmt
// CHECK: %[[NEXT:.*]] = py.invoke_indirect %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-NOT: %{{[[:alnum:]]+}}
// CHECK-NEXT: label ^[[HAPPY_PATH:[[:alnum:]]+]]
// CHECK: ^[[HAPPY_PATH]]
// CHECK: py.list.append %[[LIST]], %[[NEXT]]
// ...
// CHECK: py.list.append %[[LIST]], %[[FOUR]]
// CHECK: %[[TUPLE:.*]] = py.list.toTuple %[[LIST]]
// CHECK: return %[[TUPLE]]

