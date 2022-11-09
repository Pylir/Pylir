# REQUIRES: system-darwin

# RUN: env -u SDKROOT pylir --target x86_64-apple-darwin %s -### 2>&1 | FileCheck %s

# CHECK: -syslibroot{{.*MacOSX[0-9\.]*\.sdk[[:blank:]]}}
# CHECK: -L{{.*}}lib{{[/\\]+}}clang{{[/\\]+[0-9]+(\.[0-9]+(\.[0-9]+)?)?[/\\]+}}lib{{[/\\]+}}darwin
