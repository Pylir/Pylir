# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-DAG: #[[$ITER:.*]] = #py.globalValue<builtins.iter,
# CHECK-DAG: #[[$NEXT:.*]] = #py.globalValue<builtins.next,
# CHECK-DAG: #[[$PRINT:.*]] = #py.globalValue<builtins.print,
# CHECK-DAG: #[[$STOP:.*]] = #py.globalValue<builtins.StopIteration,

for i in (3, 5, 7):
    print(i)

# CHECK: %[[THREE:.*]] = constant(#py.int<3>)
# CHECK: %[[FIVE:.*]] = constant(#py.int<5>)
# CHECK: %[[SEVEN:.*]] = constant(#py.int<7>)
# CHECK: %[[TUPLE:.*]] = makeTuple (%[[THREE]], %[[FIVE]], %[[SEVEN]])
# CHECK: %[[ITER_F:.*]] = constant(#[[$ITER]])
# CHECK: %[[ARGS:.*]] = makeTuple (%[[TUPLE]])
# CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
# CHECK: %[[ITER:.*]] = call @pylir__call__(%[[ITER_F]], %[[ARGS]], %[[DICT]])
# CHECK: cf.br ^[[COND:[[:alnum:]]+]]

# CHECK: ^[[COND]]:
# CHECK: %[[NEXT_F:.*]] = constant(#[[$NEXT]])
# CHECK: %[[ARGS:.*]] = makeTuple (%[[ITER]])
# CHECK: %[[ITEM:.*]] = invoke @pylir__call__(%[[NEXT_F]], %[[ARGS]], %[[DICT]])
# CHECK-NEXT: label ^[[ASSIGN:.*]] unwind ^[[EXIT:[[:alnum:]]+]]

# CHECK: ^[[ASSIGN]]:
# CHECK: store %[[ITEM]] : !py.dynamic into @i$handle
# CHECK: cf.br ^[[BODY:[[:alnum:]]+]]

# CHECK: ^[[BODY]]:
# CHECK: %[[PRINT:.*]] = constant(#[[$PRINT]])
# CHECK: %[[ITEM:.*]] = load @i$handle
# CHECK: %[[ARGS:.*]] = makeTuple (%[[ITEM]])
# CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
# CHECK: call @pylir__call__(%[[PRINT]], %[[ARGS]], %[[DICT]])
# CHECK: cf.br ^[[COND]]

# CHECK: ^[[EXIT]](%[[EXC:.*]]: !py.dynamic loc({{.*}})):
# CHECK: %[[STOP_ITER:.*]] = constant(#[[$STOP]])
# CHECK: %[[EXC_TYPE:.*]] = typeOf %[[EXC]]
# CHECK: %[[IS:.*]] = is %[[STOP_ITER]], %[[EXC_TYPE]]
# CHECK: cf.cond_br %[[IS]], ^[[END:.*]], ^[[RERAISE:[[:alnum:]]+]]

# CHECK: ^[[RERAISE]]:
# CHECK: raise %[[EXC]]

# CHECK: ^[[END]]:
