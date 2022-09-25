# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

for i in (3, 5, 7):
    print(i)

# CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK: %[[FIVE:.*]] = py.constant(#py.int<5>)
# CHECK: %[[SEVEN:.*]] = py.constant(#py.int<7>)
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[THREE]], %[[FIVE]], %[[SEVEN]])
# CHECK: %[[ITER_F:.*]] = py.constant(#py.ref<@builtins.iter>)
# CHECK: %[[ARGS:.*]] = py.makeTuple (%[[TUPLE]])
# CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
# CHECK: %[[ITER:.*]] = py.call @pylir__call__(%[[ITER_F]], %[[ARGS]], %[[DICT]])
# CHECK: cf.br ^[[COND:[[:alnum:]]+]]

# CHECK: ^[[COND]]:
# CHECK: %[[NEXT_F:.*]] = py.constant(#py.ref<@builtins.next>)
# CHECK: %[[ARGS:.*]] = py.makeTuple (%[[ITER]])
# CHECK: %[[ITEM:.*]] = py.invoke @pylir__call__(%[[NEXT_F]], %[[ARGS]], %[[DICT]])
# CHECK-NEXT: label ^[[ASSIGN:.*]] unwind ^[[EXIT:[[:alnum:]]+]]

# CHECK: ^[[ASSIGN]]:
# CHECK: py.store %[[ITEM]] : !py.dynamic into @i$handle
# CHECK: cf.br ^[[BODY:[[:alnum:]]+]]

# CHECK: ^[[BODY]]:
# CHECK: %[[PRINT:.*]] = py.constant(#py.ref<@builtins.print>)
# CHECK: %[[ITEM:.*]] = py.load @i$handle
# CHECK: %[[ARGS:.*]] = py.makeTuple (%[[ITEM]])
# CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
# CHECK: py.call @pylir__call__(%[[PRINT]], %[[ARGS]], %[[DICT]])
# CHECK: cf.br ^[[COND]]

# CHECK: ^[[EXIT]](%[[EXC:.*]]: !py.dynamic loc({{.*}})):
# CHECK: %[[STOP_ITER:.*]] = py.constant(#py.ref<@builtins.StopIteration>)
# CHECK: %[[EXC_TYPE:.*]] = py.typeOf %[[EXC]]
# CHECK: %[[IS:.*]] = py.is %[[STOP_ITER]], %[[EXC_TYPE]]
# CHECK: cf.cond_br %[[IS]], ^[[END:.*]], ^[[RERAISE:[[:alnum:]]+]]

# CHECK: ^[[RERAISE]]:
# CHECK: py.raise %[[EXC]]

# CHECK: ^[[END]]:
