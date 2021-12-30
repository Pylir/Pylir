# UNSUPPORTED: system-windows
# TODO: An uncaught exception seems to be super flaky on Windows.
#       Re-enable the test if we had a bug and fixed it or we have support for
#       Catching exceptions

# RUN: pylir %s -o %t
# RUN: not %t
a
# TODO: Check the Name Error, not just that it compiles and crashes
