# RUN: not pylir -emit-llvm -emit-mlir 2>&1 | FileCheck %s --check-prefix=EMITS
# EMITS: cannot emit LLVM IR and MLIR IR at the same time

# RUN: pylir -emit-llvm -fsyntax-only 2>&1 | FileCheck %s \
# RUN: --check-prefix=SYNTAX_ONLY_LLVM
# SYNTAX_ONLY_LLVM: LLVM IR won't be emitted when only checking syntax

# RUN: pylir -emit-mlir -fsyntax-only 2>&1 | FileCheck %s \
# RUN: --check-prefix=SYNTAX_ONLY_MLIR
# SYNTAX_ONLY_MLIR: MLIR IR won't be emitted when only checking syntax

# RUN: pylir -S -fsyntax-only 2>&1 | FileCheck %s \
# RUN: --check-prefix=SYNTAX_ONLY_ASSEMBLY
# SYNTAX_ONLY_ASSEMBLY: Assembly won't be emitted when only checking syntax

# RUN: not pylir -c "x = ''" --target=wadawdagwhdawzgdwa -o - 2>&1 | \
# RUN: FileCheck %s --check-prefix=UNKNOWN_TARGET
# UNKNOWN_TARGET: could not find target 'wadawdagwhdawzgdwa'

# RUN: not pylir -c "x = ''" -Og -o - 2>&1 | \
# RUN: FileCheck %s --check-prefix=INVALID_OPT
# INVALID_OPT: invalid optimization level '-Og'
