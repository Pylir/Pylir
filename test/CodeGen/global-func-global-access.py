# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$MODULE:.*]] = #py.globalValue<__main__,

a = 3

# CHECK: globalFunc @__main__.test{{.*}}(
@pylir.intr.const_export
def test():
    # CHECK: %[[ITEM:.*]] = module_getAttr #[[$MODULE]]["a"]
    # CHECK: return %[[ITEM]]
    return a
