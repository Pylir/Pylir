# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK: py.dict_setItem %{{.*}}[{{.*}}] to %[[THREE]]
(z := 3)
