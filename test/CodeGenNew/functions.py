# RUN: pylir %s -Xnew-codegen -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: pyHIR.init "__main__"

def test1():
    pass


# CHECK: func "test1"() {
# CHECK-NEXT: %[[REF:.*]] = builtinsRef @builtins.None
# CHECK-NEXT: return %[[REF]]

def test2(arg, arg2, /):
    pass


# CHECK: func "test2"(%{{.*}}, %{{.*}})

def test3(arg, arg2, *restArg):
    pass


# CHECK: func "test3"(%{{.*}} "arg", %{{.*}} "arg2", *%{{.*}})

def test4(*, arg, arg2):
    pass


# CHECK: func "test4"(%{{.*}} only "arg", %{{.*}} only "arg2")

def test5(arg, arg2, **m):
    pass

# CHECK: func "test5"(%{{.*}} "arg", %{{.*}} "arg2", **%{{.*}})
