# RUN: pylir %s -emit-pylir -Xno-builtins -o - -S | FileCheck %s

# CHECK: #[[$MAIN_SEQ_ITER_INIT:.*]] = #py.globalValue<__main__.SeqIter.__init__,
# CHECK: #[[$MAIN_SEQ_ITER:.*]] = #py.globalValue<__main__.SeqIter,
# CHECK-SAME: const
# CHECK-SAME: initializer = #py.type<
# CHECK-SAME: mro_tuple = #py.tuple<(#py.globalValue<__main__.SeqIter>, #py.globalValue<builtins.object>)>
# CHECK-SAME: instance_slots = <(#py.str<"__seq">, #py.str<"__i">)>
# CHECK-SAME: slots = {__init__ = #py.globalValue<__main__.SeqIter.__init__,
# CHECK-SAME:          __name__ = #py.str<"SeqIter">}

# CHECK: #py.globalValue<__main__.function,
# CHECK-SAME: initializer = #py.type<
# CHECK-SAME: instance_slots = <({{.*}})>,

# CHECK-LABEL: init "__main__"

@pylir.intr.const_export
class SeqIter:
    __slots__ = ("__seq", "__i")

    def __init__(self, seq) -> None:
        pass


# CHECK: %[[MAIN_SEQ_ITER:.*]] = py.constant(#[[$MAIN_SEQ_ITER]])
# CHECK: py.dict_setItem %{{.*}}[{{.*}}] to %[[MAIN_SEQ_ITER]]

# CHECK: globalFunc @{{.*}}__init__{{.*}}(

# CHECK: py.external @__main__.SeqIter, #[[$MAIN_SEQ_ITER]]

@pylir.intr.const_export
class function:
    __slots__ = pylir.intr.function.__slots__
