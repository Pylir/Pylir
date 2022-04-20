# Copyright 2022 Markus Böck
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_interface name)
    set(LLVM_TARGET_DEFINITIONS ${name}.td)
    mlir_tablegen(${name}.h.inc -gen-op-interface-decls)
    mlir_tablegen(${name}.cpp.inc -gen-op-interface-defs)
    add_public_tablegen_target(Pylir${name}IncGen)
    add_dependencies(mlir-generic-headers Pylir${name}IncGen)

    add_library(Pylir${name} ${name}.cpp)
    add_dependencies(Pylir${name} Pylir${name}IncGen)
    target_link_libraries(Pylir${name} PUBLIC MLIRIR)
endfunction()

add_interface(CaptureInterface)
add_interface(MemoryFoldInterface)
