# RUN: env SDKROOT=%S/Inputs/MacOSX10.14.sdk pylir --target x86_64-apple-darwin %s -### 2>&1 | FileCheck %s

# CHECK: -syslibroot{{.*}}Inputs{{[/\\]+}}MacOSX10.14.sdk{{[[:blank:]]}}
