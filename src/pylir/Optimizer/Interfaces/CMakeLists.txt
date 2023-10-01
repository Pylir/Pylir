# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_interface(Attr AttrVerifyInterface LIBRARY LIB_PREFIX Pylir)
add_pylir_interface(OP CaptureInterface LIBRARY LIB_PREFIX Pylir)
add_pylir_interface(OP ConditionalBranchInterface LIBRARY LIB_PREFIX Pylir)
add_pylir_interface(OP MemoryFoldInterface LIBRARY LIB_PREFIX Pylir)

add_pylir_interface(OP SROAOpInterfaces FILE SROAInterfaces LIB_PREFIX Pylir)
add_pylir_interface(ATTR SROAAttrInterfaces FILE SROAInterfaces LIB_PREFIX Pylir)
add_library(PylirSROAInterfaces
  SROAInterfaces.cpp
)
add_dependencies(PylirSROAInterfaces
  PylirSROAAttrInterfacesIncGen
  PylirSROAOpInterfacesIncGen
)
target_link_libraries(PylirSROAInterfaces
  PUBLIC
  MLIRIR
)
