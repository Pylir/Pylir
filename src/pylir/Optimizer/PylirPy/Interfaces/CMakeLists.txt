# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_interface name)
    set(LLVM_TARGET_DEFINITIONS ${name}.td)
    mlir_tablegen(${name}.h.inc -gen-op-interface-decls)
    mlir_tablegen(${name}.cpp.inc -gen-op-interface-defs)
    add_public_tablegen_target(PylirPy${name}IncGen)
    add_dependencies(mlir-generic-headers PylirPy${name}IncGen)

    add_library(PylirPy${name} ${name}.cpp)
    add_dependencies(PylirPy${name} PylirPy${name}IncGen)
    target_link_libraries(PylirPy${name} PUBLIC MLIRIR)
endfunction()

add_interface(TypeRefineableInterface)
add_interface(TypeFoldInterface)
add_interface(ExceptionHandlingInterface)
add_interface(ObjectFromTypeObjectInterface)
