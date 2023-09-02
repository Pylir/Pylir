// RUN: pylir-opt -symbol-dce %s | FileCheck %s

// CHECK: py.func private @bar()
py.func private @bar() {
    return
}

py.external @foo, #py.globalValue<foo, const, initializer = #py.function<@bar>>
