# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

# CHECK-LABEL: init "__main__"
# CHECK: %[[TEST1:.*]] = func "__main__.test1"
def test1():
    # CHECK: %[[NESTED:.*]] = func "__main__.test1.<locals>.nested"
    def nested():
        pass

    other = nested

    # CHECK: func "__main__.test1.<locals>.has_default"(%[[A:.*]] "a" = %[[NESTED]])
    def has_default(a=other):
        # CHECK: func "__main__.test1.<locals>.has_default.<locals>.parameter_use"(%{{.*}} "b" = %[[A]])
        def parameter_use(b=a):
            pass


# CHECK: %[[STR:.*]] = py.constant(#py.str<"test1">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: py.dict_setItem %{{.*}}[%[[STR]] hash(%[[HASH]])] to %[[TEST1]]

# CHECK: %[[STR:.*]] = py.constant(#py.str<"test1">)
# CHECK: %[[HASH:.*]] = py.str_hash %[[STR]]
# CHECK: %[[TEST1:.*]] = py.dict_tryGetItem %{{.*}}[%[[STR]] hash(%[[HASH]])]
# CHECK: func "__main__.test2"(%{{.*}} "arg" = %[[TEST1]])

def test2(arg=test1):
    pass
