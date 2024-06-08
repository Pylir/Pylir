# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

a = 3

# CHECK: globalFunc @__main__.test{{.*}}(
@pylir.intr.const_export
def test():
    # CHECK: %[[DIC:.*]] = py.constant(#__main__$dict)
    # CHECK: %[[STR:.*]] = py.constant(#py.str<"a">)
    # CHECK: %[[ITEM:.*]] = py.dict_tryGetItem %[[DIC]][%[[STR]] hash(%{{.*}})]
    # CHECK: return %[[ITEM]]
    return a
