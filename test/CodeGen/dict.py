# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$HASH:.*]] = #py.globalValue<builtins.hash,

# CHECK-LABEL: func @__init__

{}

# CHECK: makeDict
# CHECK-SAME: ()

{7: 5}

# CHECK: %[[OP1:.*]] = constant(#py.int<7>)
# CHECK: %[[OP2:.*]] = constant(#py.int<5>)
# CHECK: %[[TUPLE:.*]] = makeTuple (%[[OP1]])
# CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
# CHECK: %[[REF:.*]] = constant(#[[$HASH]])
# CHECK: %[[RES:.*]] = call @pylir__call__(%[[REF]], %[[TUPLE]], %[[DICT]])
# CHECK: %[[HASH:.*]] = int_toIndex %[[RES]]
# CHECK: makeDict
# CHECK-SAME: %[[OP1]] hash(%[[HASH]]) : %[[OP2]]

{7: 5, 5: 3}

# CHECK: %[[OP1:.*]] = constant(#py.int<7>)
# CHECK: %[[OP2:.*]] = constant(#py.int<5>)
# CHECK: %[[TUPLE:.*]] = makeTuple (%[[OP1]])
# CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
# CHECK: %[[REF:.*]] = constant(#[[$HASH]])
# CHECK: %[[RES:.*]] = call @pylir__call__(%[[REF]], %[[TUPLE]], %[[DICT]])
# CHECK: %[[HASH1:.*]] = int_toIndex %[[RES]]
# CHECK: %[[OP3:.*]] = constant(#py.int<5>)
# CHECK: %[[OP4:.*]] = constant(#py.int<3>)
# CHECK: %[[TUPLE:.*]] = makeTuple (%[[OP3]])
# CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
# CHECK: %[[REF:.*]] = constant(#[[$HASH]])
# CHECK: %[[RES:.*]] = call @pylir__call__(%[[REF]], %[[TUPLE]], %[[DICT]])
# CHECK: %[[HASH3:.*]] = int_toIndex %[[RES]]
# CHECK: makeDict
# CHECK-SAME: %[[OP1]] hash(%[[HASH1]]) : %[[OP2]], %[[OP3]] hash(%[[HASH3]]) : %[[OP4]]

{**{}, 3: 5}

# CHECK: %[[OP1:.*]] = makeDict ()
# CHECK: %[[OP2:.*]] = constant(#py.int<3>)
# CHECK: %[[OP3:.*]] = constant(#py.int<5>)
# CHECK: %[[TUPLE:.*]] = makeTuple (%[[OP2]])
# CHECK: %[[DICT:.*]] = constant(#py.dict<{}>)
# CHECK: %[[REF:.*]] = constant(#[[$HASH]])
# CHECK: %[[RES:.*]] = call @pylir__call__(%[[REF]], %[[TUPLE]], %[[DICT]])
# CHECK: %[[HASH:.*]] = int_toIndex %[[RES]]
# CHECK: makeDict
# CHECK-SAME: **%[[OP1]], %[[OP2]] hash(%[[HASH]]) : %[[OP3]]
