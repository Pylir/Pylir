# RUN: not pylir -Xsanitize=address,thread -### %s 2>&1 | FileCheck %s --check-prefix=ASAN_TSAN

# ASAN_TSAN: 'address' and 'thread' sanitizers are incompatible with each other

# RUN: not pylir -Xsanitize=thing -### %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN_SAN

# UNKNOWN_SAN: unknown sanitizer 'thing'
