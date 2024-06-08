# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$MAIN_FOO:.*]] = #py.globalValue<__main__.foo,
# CHECK-SAME: const
# CHECK-SAME: initializer = #py.function<@[[$MAIN_FOO_SYMBOL:[0-9a-zA-Z_.]+]]
# CHECK-SAME: qual_name = #py.str<"__main__.foo">
# CHECK-SAME: defaults = #py.tuple<(#py.int<3>)>
# CHECK-SAME: kw_defaults = #py.dict<{#py.str<"b"> to #py.globalValue<builtins.None{{.*}}>}>

# CHECK-LABEL: init "__main__"
# CHECK: %[[FOO:.*]] = py.constant(#[[$MAIN_FOO]])
# CHECK: py.dict_setItem %{{.*}}[{{.*}}] to %[[FOO]]
@pylir.intr.const_export
def foo(a=3, *, b=None):
    return a + b

# CHECK: globalFunc @[[$MAIN_FOO_SYMBOL]](%{{[[:alnum:]]+}}, %[[ARG0:.*]] "a" has_default, %[[ARG1:.*]] only "b" has_default) {
# CHECK: binOp %[[ARG0]] __add__ %[[ARG1]]

# CHECK: py.external @__main__.foo, #[[$MAIN_FOO]]
