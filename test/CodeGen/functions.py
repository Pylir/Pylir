# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK: #[[$NONE:.*]] = #py.globalValue<builtins.None{{>|,}}

# CHECK-LABEL: pyHIR.init "__main__"

def test1():
    pass


# CHECK: func "__main__.test1"() {
# CHECK-NEXT: %[[REF:.*]] = py.constant(#[[$NONE]])
# CHECK-NEXT: return %[[REF]]

def test2(arg, arg2, /):
    def nested():
        pass

    pass


# CHECK: func "__main__.test2"(%{{.*}}, %{{.*}})
# CHECK: func "__main__.test2.<locals>.nested"()

def test3(arg, arg2, *restArg):
    pass


# CHECK: func "__main__.test3"(%{{.*}} "arg", %{{.*}} "arg2", *%{{.*}})

def test4(*, arg, arg2):
    pass


# CHECK: func "__main__.test4"(%{{.*}} only "arg", %{{.*}} only "arg2")

def test5(arg, arg2, **m):
    pass


# CHECK: func "__main__.test5"(%{{.*}} "arg", %{{.*}} "arg2", **%{{.*}})

def test6(arg=3, /):
    pass

# CHECK: %[[THREE:.*]] = py.constant(#py.int<3>)
# CHECK: func "__main__.test6"(%{{.*}} = %[[THREE]])
