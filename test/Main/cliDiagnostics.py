# RUN: not pylir -emit-llvm -emit-mlir 2>&1 | FileCheck %s \
# RUN: --check-prefix=EMITS_LLVM_MLIR
# EMITS_LLVM_MLIR: cannot emit LLVM IR and MLIR IR at the same time

# RUN: not pylir -emit-llvm -emit-pylir 2>&1 | FileCheck %s \
# RUN: --check-prefix=EMITS_LLVM_PYLIR
# EMITS_LLVM_PYLIR: cannot emit LLVM IR and Pylir IR at the same time

# RUN: not pylir -emit-mlir -emit-pylir 2>&1 | FileCheck %s \
# RUN: --check-prefix=EMITS_MLIR_PYLIR
# EMITS_MLIR_PYLIR: cannot emit MLIR IR and Pylir IR at the same time

# RUN: pylir -emit-llvm -fsyntax-only 2>&1 | FileCheck %s \
# RUN: --check-prefix=SYNTAX_ONLY_LLVM
# SYNTAX_ONLY_LLVM: LLVM IR won't be emitted when only checking syntax

# RUN: pylir -emit-mlir -fsyntax-only 2>&1 | FileCheck %s \
# RUN: --check-prefix=SYNTAX_ONLY_MLIR
# SYNTAX_ONLY_MLIR: MLIR IR won't be emitted when only checking syntax

# RUN: pylir -emit-pylir -fsyntax-only 2>&1 | FileCheck %s \
# RUN: --check-prefix=SYNTAX_ONLY_PYLIR
# SYNTAX_ONLY_PYLIR: Pylir IR won't be emitted when only checking syntax

# RUN: not pylir -emit-llvm 2>&1 | FileCheck %s \
# RUN: --check-prefix=LLVM_LINKING
# LLVM_LINKING: cannot emit LLVM IR when linking

# RUN: not pylir -emit-mlir 2>&1 | FileCheck %s \
# RUN: --check-prefix=MLIR_LINKING
# MLIR_LINKING: cannot emit MLIR IR when linking

# RUN: not pylir -emit-pylir 2>&1 | FileCheck %s \
# RUN: --check-prefix=PYLIR_LINKING
# PYLIR_LINKING: cannot emit Pylir IR when linking

# RUN: pylir -S -fsyntax-only 2>&1 | FileCheck %s \
# RUN: --check-prefix=SYNTAX_ONLY_ASSEMBLY
# SYNTAX_ONLY_ASSEMBLY: Assembly won't be emitted when only checking syntax

# RUN: pylir -c -fsyntax-only 2>&1 | FileCheck %s \
# RUN: --check-prefix=SYNTAX_ONLY_OBJECT
# SYNTAX_ONLY_OBJECT: Object file won't be emitted when only checking syntax

# RUN: not pylir %s --target=wadawdagwhdawzgdwa -o - 2>&1 | \
# RUN: FileCheck %s --check-prefix=UNKNOWN_TARGET
# RUN: not pylir %s --target wadawdagwhdawzgdwa -o - 2>&1 | \
# RUN: FileCheck %s --check-prefix=UNKNOWN_TARGET
# UNKNOWN_TARGET: could not find target 'wadawdagwhdawzgdwa'

# RUN: not pylir %s -Og -o - 2>&1 | \
# RUN: FileCheck %s --check-prefix=INVALID_OPT
# INVALID_OPT: invalid optimization level '-Og'

# RUN: not pylir 2>&1 | FileCheck %s --check-prefix=NO_INPUT
# NO_INPUT: no input file

# RUN: not pylir test.py test2.py 2>&1 | FileCheck %s --check-prefix=MORE_INPUT
# MORE_INPUT: expected only one input file
