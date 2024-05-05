# RUN: pylir %s -emit-pylir -o - -c -S | FileCheck %s

# CHECK-LABEL: init "__main__"

def x(*args, **kwargs):
    pass


# CHECK: %[[X:.*]] = module_getAttr #{{.*}}["x"]
# CHECK: %[[FIVE:.*]] = py.constant(#py.int<5>)
# CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK: %[[SEVEN:.*]] = py.constant(#py.int<7>)
# CHECK: %[[EIGHT:.*]] = py.constant(#py.int<8>)
# CHECK: call %[[X]](%[[FIVE]], "k"=%[[THREE]], *%[[SEVEN]], **%[[EIGHT]])
x(5, k=3, *7, **8)
