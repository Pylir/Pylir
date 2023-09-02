// RUN: pylir-opt %s | pylir-opt | FileCheck %s

// CHECK: #[[ATTR:.*]] = #py.globalValue<builtins.str, const, initializer = #py.type>

// CHECK: py.external @builtins.str, #[[ATTR]]

py.external @builtins.str, #py.globalValue<builtins.str, const, initializer = #py.type<mro_tuple = #py.tuple<()>>>
