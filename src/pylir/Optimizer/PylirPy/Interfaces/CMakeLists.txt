# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_pylir_interface(OP ExceptionHandlingInterface LIBRARY LIB_PREFIX PylirPy)
add_pylir_interface(OP KnownTypeObjectInterface LIBRARY LIB_PREFIX PylirPy)
add_pylir_interface(OP CopyObjectInterface LIBRARY LIB_PREFIX PylirPy)
add_pylir_interface(OP OnlyReadsValueInterface LIBRARY LIB_PREFIX PylirPy)
