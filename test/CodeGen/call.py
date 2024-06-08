# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

# CHECK-LABEL: init "__main__"
# CHECK: %[[GLOBALS:.*]] = py.constant(#__main__$dict)

def x(*args, **kwargs):
    pass


# CHECK: py.dict_setItem

# CHECK: %[[STR:.*]] = py.constant(#py.str<"x">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: %[[X:.*]] = py.dict_tryGetItem %{{.*}}[%[[STR]] hash(%[[HASH]])]
# CHECK: %[[FIVE:.*]] = py.constant(#py.int<5>)
# CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK: %[[SEVEN:.*]] = py.constant(#py.int<7>)
# CHECK: %[[EIGHT:.*]] = py.constant(#py.int<8>)
# CHECK: call %[[X]](%[[FIVE]], "k"=%[[THREE]], *%[[SEVEN]], **%[[EIGHT]])
x(5, k=3, *7, **8)
