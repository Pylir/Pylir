// RUN: pylir-opt %s -convert-pylirMem-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type<slots: #py.slots<{"__slots__" to #py.tuple<(#py.str<"__slots__">,#py.str<"__eq__">,#py.str<"__hash__">)>}>>

py.globalValue const @builtins.object = #py.type // stub
py.globalValue const @builtins.str = #py.type // stub
py.globalValue const @builtins.tuple = #py.type // stub
