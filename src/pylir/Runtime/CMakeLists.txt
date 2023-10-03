# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if (CMAKE_CXX_COMPILER_TARGET)
  set(TARGET_TRIPLE ${CMAKE_CXX_COMPILER_TARGET})
else ()
  set(TARGET_TRIPLE ${LLVM_TARGET_TRIPLE})
endif ()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/pylir/${TARGET_TRIPLE}")

# Collects all non-imported targets that 'target' links against transitively into 'listvar', including 'target' itself.
function(collect_libs target listvar)
  set(worklist ${${listvar}})
  list(APPEND worklist ${target})
  
  get_target_property(subDeps ${target} LINK_LIBRARIES)
  foreach (dep IN LISTS subDeps)
    if (NOT TARGET ${dep})
      continue()
    endif ()
    
    get_target_property(depImported ${dep} IMPORTED)
    if (NOT dep IN_LIST worklist AND NOT ${depImported})
      collect_libs(${dep} worklist)
    endif ()
  endforeach ()
  set(${listvar} "${worklist}" PARENT_SCOPE)
endfunction()

add_library(PylirRuntime STATIC
  CAPI/API.cpp
  ExceptionHandling/ExceptionHandling.cpp
  GC/Globals.cpp
  GC/Stack.cpp
  Modules/SysModule.cpp
  Objects/Objects.cpp
  Objects/Support.cpp
  Util/Pages.cpp
)
target_link_libraries(PylirRuntime
  PUBLIC
  PylirSupport
)

if (MSVC)
  target_link_libraries(PylirRuntime
    PUBLIC
    unwind_static
  )
endif ()

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  find_path(LIBUNWIND_INCLUDE_DIR libunwind.h REQUIRED)
  find_library(LIBUNWIND_LIB unwind REQUIRED)
  target_link_libraries(PylirRuntime
    PRIVATE
    ${LIBUNWIND_LIB}
  )
  target_include_directories(PylirRuntime PRIVATE ${LIBUNWIND_INCLUDE_DIR})
endif ()

collect_libs(PylirRuntime LIBS)
# First one is always 'PylirRuntime' itself.
list(POP_FRONT LIBS)
list(TRANSFORM LIBS REPLACE "(.+)" "$<TARGET_OBJECTS:\\1>")
target_sources(PylirRuntime PRIVATE ${LIBS})

add_subdirectory(MarkAndSweep)

add_library(PylirRuntimeMain STATIC PylirRuntimeMain.cpp)

