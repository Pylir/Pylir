# RUN: pylir %s -o %t
# RUN: not %t
a
# TODO: Check the Name Error, not just that it compiles and crashes
