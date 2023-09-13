# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(sources
  builtins.py
  pylir/__init__.pyi
  pylir/intr/__init__.pyi
  pylir/intr/BaseException/__init__.pyi
  pylir/intr/bool/__init__.pyi
  pylir/intr/dict/__init__.pyi
  pylir/intr/function/__init__.pyi
  pylir/intr/int/__init__.pyi
  pylir/intr/intr/__init__.pyi
  pylir/intr/list/__init__.pyi
  pylir/intr/object/__init__.pyi
  pylir/intr/str/__init__.pyi
  pylir/intr/tuple/__init__.pyi
  pylir/intr/type/__init__.pyi
)

set(outputs)
foreach (f ${sources})
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/${f})
  set(dst ${CMAKE_BINARY_DIR}/lib/${f})
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying pylirs ${f}..."
  )
  list(APPEND outputs ${dst})
endforeach ()

add_custom_target(pylir-stdlib ALL DEPENDS ${outputs} SOURCES ${sources})
