# RUN: rm -f %t
# RUN: pylir %s -c -o %t
# RUN: ls %t

# RUN: rm -f %t
# RUN: pylir %s -emit-mlir -c -o %t
# RUN: ls %t

# RUN: rm -f %t
# RUN: pylir %s -emit-pylir -c -o %t
# RUN: ls %t

# RUN: rm -f %t
# RUN: pylir %s -emit-llvm -c -o %t
# RUN: ls %t

# RUN: rm -f %t
# RUN: pylir %s -S -o %t
# RUN: ls %t

# RUN: rm -f %t
# RUN: pylir %s -S -emit-llvm -o %t
# RUN: ls %t

# Writes to stdout instead of a file
# RUN: rm -f %t
# RUN: pylir %s -S -emit-llvm -o -
# RUN: not ls %t

# File already exists, should not error and just overwrite
# RUN: rm -f %t
# RUN: touch %t
# RUN: pylir %s -S -emit-llvm -o %t
# RUN: grep __init__ %t
