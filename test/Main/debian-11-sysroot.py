# RUN: pylir %s --sysroot=%S/Inputs/debian-11-sysroot -### --target=x86_64-unknown-linux-gnu 2>&1 | FileCheck %s

# CHECK: -L{{.*}}debian-11-sysroot{{[/\\]+}}usr{{[/\\]+}}lib{{[[:space:]]}}
# CHECK: -L{{.*}}debian-11-sysroot{{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}gcc{{[/\\]+}}x86_64-linux-gnu{{[/\\]+}}10{{[[:space:]]}}
# CHECK: -L{{.*}}debian-11-sysroot{{[/\\]+}}lib{{[/\\]+}}x86_64-linux-gnu{{[[:space:]]}}
# CHECK: -L{{.*}}debian-11-sysroot{{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}x86_64-linux-gnu{{[[:space:]]}}

