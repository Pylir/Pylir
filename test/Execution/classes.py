#  Licensed under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#  Licensed under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s

class Test:
    pass


t = Test()
# CHECK: <__main__.Test object at {{[[:alnum:]]+}}>
print(t)
