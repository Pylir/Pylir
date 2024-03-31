# RUN: rm -f %t
# RUN: pylir %s -c -o %t
# RUN: ls %t

# RUN: rm -f %t
# RUN: pylir %s -emit-pylir -o %t -S
# RUN: grep "init \"__main__\"" %t

# RUN: rm -f %t
# RUN: pylir %s -emit-pylir -o %t -c
# RUN: od -x -N 4 %t | grep "4c4d *52ef"

# RUN: rm -f %t
# RUN: pylir %s -emit-llvm -c -o %t
# TODO: Why does this differ on Mac?
# RUN: od -x -N 4 %t | grep "\(4342 *dec0\)\|\(c0de *0b17\)"

# RUN: rm -f %t
# RUN: pylir %s -S -o %t
# RUN: ls %t

# RUN: rm -f %t
# RUN: pylir %s -emit-llvm -o %t -S
# RUN: grep "define .* @__init__()" %t

# Last of the `-emit-*` options is actually used
# RUN: rm -f %t
# RUN: pylir %s -emit-pylir -emit-llvm -o %t
# RUN: grep "define .* @__init__()" %t

# Writes to stdout instead of a file
# RUN: rm -f %t
# RUN: pylir %s -emit-llvm -o -
# RUN: not ls %t

# File already exists, should not error and just overwrite
# RUN: rm -f %t
# RUN: touch %t
# RUN: pylir %s -emit-llvm -o %t
# RUN: grep "define .* @__init__()" %t
