// RUN: pylir %s -o - -S -emit-llvm | FileCheck %s
// RUN: pylir-opt %s -o %t.mlirbc -emit-bytecode
// RUN: pylir %t.mlirbc -o - -S -emit-llvm | FileCheck %s

#const = #py.globalValue<"const$", const, initializer = #py.tuple<(#py.str<"__slots__">)>>
#builtins_type = #py.globalValue<builtins.type, initializer = #py.type<slots = {__slots__ = #const}>>
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
#builtins_function = #py.globalValue<builtins.function, initializer = #py.type>
#builtins_dict = #py.globalValue<builtins.dict, initializer = #py.type>

py.external @builtins.type, #builtins_type
py.external @builtins.tuple, #builtins_tuple
py.external @builtins.str, #builtins_str
py.external @builtins.function, #builtins_function
py.external @builtins.dict, #builtins_dict

py.func @foo(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = typeOf %arg0
    %1 = typeOf %0
    %c0 = arith.constant 0 : index
    %2 = getSlot %0[%c0]
    return %2 : !py.dynamic
}

// CHECK-LABEL: define {{.*}} ptr addrspace({{[0-9]+}}) @foo(ptr addrspace({{[0-9]+}}) %{{.*}})
