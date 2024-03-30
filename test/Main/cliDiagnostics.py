# RUN: pylir -emit-llvm -fsyntax-only %s 2>&1 | FileCheck %s \
# RUN: --check-prefix=SYNTAX_ONLY_LLVM
# SYNTAX_ONLY_LLVM: LLVM IR won't be emitted when only checking syntax

# RUN: pylir -emit-pylir -fsyntax-only %s 2>&1 | FileCheck %s \
# RUN: --check-prefix=SYNTAX_ONLY_PYLIR
# SYNTAX_ONLY_PYLIR: Pylir IR won't be emitted when only checking syntax

# RUN: pylir -S -fsyntax-only %s 2>&1 | FileCheck %s \
# RUN: --check-prefix=SYNTAX_ONLY_ASSEMBLY
# SYNTAX_ONLY_ASSEMBLY: Assembly won't be emitted when only checking syntax

# RUN: pylir -c -fsyntax-only %s 2>&1 | FileCheck %s \
# RUN: --check-prefix=SYNTAX_ONLY_OBJECT
# SYNTAX_ONLY_OBJECT: Object file won't be emitted when only checking syntax

# RUN: not pylir %s --target=wadawdagwhdawzgdwa -o - 2>&1 | \
# RUN: FileCheck %s --check-prefix=UNKNOWN_TARGET
# RUN: not pylir %s --target wadawdagwhdawzgdwa -o - 2>&1 | \
# RUN: FileCheck %s --check-prefix=UNKNOWN_TARGET
# UNKNOWN_TARGET: unsupported target 'wadawdagwhdawzgdwa'

# RUN: not pylir %s -Og -o - 2>&1 | \
# RUN: FileCheck %s --check-prefix=INVALID_OPT
# INVALID_OPT: invalid optimization level '-Og'

# RUN: not pylir 2>&1 | FileCheck %s --check-prefix=NO_INPUT
# NO_INPUT: no input file

# RUN: not pylir test.py test2.py 2>&1 | FileCheck %s --check-prefix=MORE_INPUT
# MORE_INPUT: expected only one input file
