# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_passes(Passes Conversion PREFIX Pylir)

add_subdirectory(PylirHIRToPylirPy)
add_subdirectory(PylirPyToPylirMem)
add_subdirectory(PylirToLLVMIR)
