# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-NOT: instance_slots = <(#py.tuple<(
