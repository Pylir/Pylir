# RUN: pylir %s -fsyntax-only -dump-ast | FileCheck --match-full-lines %s

def test4(*, arg):
    pass

# CHECK: `-function test4
# CHECK-NEXT: |-parameter arg keyword-only
# CHECK-NEXT: |-locals: arg
# CHECK-NEXT: `-suite
# CHECK-NEXT:   `-'pass' statement
