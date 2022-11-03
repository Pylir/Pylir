# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_dialect(TypeFlowIR typeFlow)
add_pylir_interface(OP TypeFlowInterfaces LIB_PREFIX PylirPy)

add_library(PylirPyAnalysis TypeFlowIR.cpp TypeFlow.cpp TypeFlowInterfaces.cpp)
target_link_libraries(PylirPyAnalysis
        PUBLIC
        MLIRIR
        PylirPyDialect
        MLIRControlFlowInterfaces
        MLIRControlFlowDialect
        MLIRCallInterfaces
        )
add_dependencies(PylirPyAnalysis TypeFlowIRIncGen)
add_dependencies(PylirPyAnalysis PylirPyTypeFlowInterfacesIncGen)
