# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: __init__

{}

# CHECK: makeDict
# CHECK-SAME: ()

{7: 5}

# CHECK: %[[OP1:.*]] = py.constant(#py.int<7>)
# CHECK: %[[OP2:.*]] = py.constant(#py.int<5>)
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[OP1]])
# CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
# CHECK: %[[REF:.*]] = py.constant(@builtins.hash)
# CHECK: %[[RES:.*]] = py.call @pylir__call__(%[[REF]], %[[TUPLE]], %[[DICT]])
# CHECK: %[[HASH:.*]] = py.int.toIndex %[[RES]]
# CHECK: makeDict
# CHECK-SAME: %[[OP1]] hash(%[[HASH]]) : %[[OP2]]

{7: 5, 5: 3}

# CHECK: %[[OP1:.*]] = py.constant(#py.int<7>)
# CHECK: %[[OP2:.*]] = py.constant(#py.int<5>)
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[OP1]])
# CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
# CHECK: %[[REF:.*]] = py.constant(@builtins.hash)
# CHECK: %[[RES:.*]] = py.call @pylir__call__(%[[REF]], %[[TUPLE]], %[[DICT]])
# CHECK: %[[HASH1:.*]] = py.int.toIndex %[[RES]]
# CHECK: %[[OP3:.*]] = py.constant(#py.int<5>)
# CHECK: %[[OP4:.*]] = py.constant(#py.int<3>)
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[OP3]])
# CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
# CHECK: %[[REF:.*]] = py.constant(@builtins.hash)
# CHECK: %[[RES:.*]] = py.call @pylir__call__(%[[REF]], %[[TUPLE]], %[[DICT]])
# CHECK: %[[HASH3:.*]] = py.int.toIndex %[[RES]]
# CHECK: makeDict
# CHECK-SAME: %[[OP1]] hash(%[[HASH1]]) : %[[OP2]], %[[OP3]] hash(%[[HASH3]]) : %[[OP4]]

{**{}, 3: 5}

# CHECK: %[[OP1:.*]] = py.makeDict ()
# CHECK: %[[OP2:.*]] = py.constant(#py.int<3>)
# CHECK: %[[OP3:.*]] = py.constant(#py.int<5>)
# CHECK: %[[TUPLE:.*]] = py.makeTuple (%[[OP2]])
# CHECK: %[[DICT:.*]] = py.constant(#py.dict<{}>)
# CHECK: %[[REF:.*]] = py.constant(@builtins.hash)
# CHECK: %[[RES:.*]] = py.call @pylir__call__(%[[REF]], %[[TUPLE]], %[[DICT]])
# CHECK: %[[HASH:.*]] = py.int.toIndex %[[RES]]
# CHECK: makeDict
# CHECK-SAME: **%[[OP1]], %[[OP2]] hash(%[[HASH]]) : %[[OP3]]
