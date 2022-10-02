# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(Util)

add_pylir_passes(Passes Transform PREFIX Pylir)

add_library(PylirTransforms LoadForwardingPass.cpp SROA.cpp FixpointPass.cpp)
add_dependencies(PylirTransforms PylirTransformPassIncGen)
target_link_libraries(PylirTransforms PUBLIC MLIRPass PRIVATE PylirAnalysis PylirMemoryFoldInterface
  PylirSROAInterfaces PylirTransformsUtils)
set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS PylirTransforms)
