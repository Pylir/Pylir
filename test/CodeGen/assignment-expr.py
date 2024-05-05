# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK: module_setAttr #{{.*}}["z"] to %[[THREE]]
(z := 3)
