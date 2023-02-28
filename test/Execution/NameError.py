# RUN: pylir %rt_link_flags %s -o %t -O3
# RUN: not %t 2>&1 | FileCheck %s

a
# CHECK: NameError:
# TODO error messagea
