# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_compile_definitions(DO_NOT_USE_WMAIN)
add_subdirectory(Catch2-3.1.0 EXCLUDE_FROM_ALL)
add_subdirectory(fmt-7.1.3 EXCLUDE_FROM_ALL)
add_subdirectory(utf8proc EXCLUDE_FROM_ALL)
add_subdirectory(libtommath EXCLUDE_FROM_ALL)
if (MSVC)
  # Workaround for cmake issue:
  # https://gitlab.kitware.com/cmake/cmake/-/commit/b41fab928551d5a71af2c16b34f9683d872818b0
  # TODO: Remove below once CMake 3.24 or higher is the minimum required version.
  set(CMAKE_ASM-ATT_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded "")
  set(CMAKE_ASM-ATT_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL "")
  set(CMAKE_ASM-ATT_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug "")
  set(CMAKE_ASM-ATT_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL "")
  set(LIBUNWIND_ENABLE_SHARED OFF)
  add_compile_definitions(__SEH__)
  add_compile_options(-fno-sanitize=all -fno-profile-instr-generate -fno-coverage-mapping)
  add_subdirectory(libunwind EXCLUDE_FROM_ALL)
endif ()

# Propagate module path set by 3rd party libraries upwards.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
