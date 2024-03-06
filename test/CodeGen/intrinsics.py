# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

import pylir.intr.object

# CHECK-LABEL: init "__main__"

def foo():
    pass


# CHECK: py.dict_setItem
# CHECK: %[[STR:.*]] = py.constant(#py.str<"foo">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: %[[FOO:.*]] = py.dict_tryGetItem %{{.*}}[%[[STR]] hash(%[[HASH]])]
# CHECK: %[[T:.*]] = py.typeOf %[[FOO]]
# CHECK: %[[STR:.*]] = py.constant(#py.str<"t">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: py.dict_setItem %{{.*}}[%[[STR]] hash(%[[HASH]])] to %[[T]]
t = pylir.intr.typeOf(foo)

# CHECK: %[[ZERO:.*]] = py.constant(#py.int<0>)
# CHECK: %[[ONE:.*]] = py.constant(#py.int<1>)
# CHECK: %[[CMP:.*]] = py.int_cmp ne %[[ZERO]], %[[ONE]]
# CHECK: %[[BOOL:.*]] = py.bool_fromI1 %[[CMP]]
pylir.intr.int.cmp("ne", 0, 1)

# CHECK: %[[ZERO:.*]] = py.constant(#py.int<0>)
# CHECK: %[[ONE:.*]] = py.constant(#py.int<1>)
# CHECK: %[[TWO:.*]] = py.constant(#py.int<2>)
# CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK: py.function_call %[[ZERO]](%[[ONE]], %[[TWO]], %[[THREE]])
pylir.intr.function.call(0, 1, 2, 3)

# CHECK: py.constant(#py.tuple<({{.*}})>)
pylir.intr.function.__slots__
