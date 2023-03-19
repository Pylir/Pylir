# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

5
# CHECK: constant(#py.int<5>)

6.5
# CHECK: constant(#py.float<6.5{{0*}}e+{{0+}}>)

'text'
# CHECK: constant(#py.str<"text">)

True

# CHECK: constant(#py.bool<True>)

False

# CHECK: constant(#py.bool<False>)

None
# CHECK: constant(#py.ref<@builtins.None>)
