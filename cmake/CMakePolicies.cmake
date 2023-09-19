# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# mlir-tblgen and llvm-tblgen cmake files still use relative paths. Changing the
# policy here leads to their dep file not being accepted by ninja anymore, and
# the targets always being out of date.
cmake_policy(SET CMP0116 OLD)
# GENERATED source property is globally visible, not just in directory.
cmake_policy(SET CMP0118 NEW)
# TARGET_* generator expressions in custom commands implicitly creating a
# dependency is deprecated.
cmake_policy(SET CMP0112 NEW)
