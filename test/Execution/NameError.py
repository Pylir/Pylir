# RUN: pylir %s -o %t
# RUN: not --crash %t
a
# TODO: Check the Name Error, not just that it compiles and crashes
