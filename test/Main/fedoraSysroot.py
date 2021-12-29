# RUN: pylir %s --sysroot=%S/Inputs/fedora-sysroot -### --target=x86_64-unknown-linux-gnu 2>&1 | FileCheck %s
# CHECK: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}..{{[/\\]+}}lib64{{[/\\]+}}crt1.o
# CHECK-SAME: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}..{{[/\\]+}}lib64{{[/\\]+}}crti.o
# CHECK-SAME: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}gcc{{[/\\]+}}x86_64-redhat-linux{{[/\\]+}}11{{[/\\]+}}crtbegin.o
# CHECK-SAME: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}gcc{{[/\\]+}}x86_64-redhat-linux{{[/\\]+}}11{{[/\\]+}}crtend.o
# CHECK-SAME: {{[/\\]+}}usr{{[/\\]+}}lib{{[/\\]+}}..{{[/\\]+}}lib64{{[/\\]+}}crtn.o
