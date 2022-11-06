# RUN: pylir %s --sysroot=%S/Inputs/ubuntu-22.04-sysroot -### --target=x86_64-unknown-linux-gnu 2>&1 | FileCheck %s

# CHECK: -L{{.*}}ubuntu-22.04-sysroot{{[/\\]+}}usr{{[/\\]+}}lib{{[[:space:]]}}
# CHECK: -L{{.*}}ubuntu-22.04-sysroot{{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}gcc{{[/\\]+}}x86_64-linux-gnu{{[/\\]+}}11{{[[:space:]]}}
# CHECK: -L{{.*}}ubuntu-22.04-sysroot{{[/\\]+}}lib{{[/\\]+}}x86_64-linux-gnu{{[[:space:]]}}
# CHECK: -L{{.*}}ubuntu-22.04-sysroot{{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}x86_64-linux-gnu{{[[:space:]]}}

