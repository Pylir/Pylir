# RUN: pylir %s -emit-pylir -o - -S | FileCheck %s

def foo():
    global x
    try:
        pass
    finally:
        x = 3


# CHECK-LABEL: @"foo$impl[0]"
# CHECK-NOT: ^{{[[:alnum:]]+}}
# CHECK: %[[THREE:.*]] = constant(#py.int<3>)
# CHECK: store %[[THREE]] : !py.dynamic into @x

def foo2():
    global x
    try:
        return 5
    finally:
        x = 3


# CHECK-LABEL: @"foo2$impl[0]"
# CHECK-NOT: ^{{[[:alnum:]]+}}
# CHECK: %[[FIVE:.*]] = constant(#py.int<5>)
# CHECK: %[[THREE:.*]] = constant(#py.int<3>)
# CHECK: store %[[THREE]] : !py.dynamic into @x
# CHECK: return %[[FIVE]]

def foo3():
    global x
    try:
        try:
            pass
        finally:
            x = 5
    finally:
        x = 3


# CHECK-LABEL: @"foo3$impl[0]"
# CHECK-NOT: ^{{[[:alnum:]]+}}
# CHECK: %[[FIVE:.*]] = constant(#py.int<5>)
# CHECK: store %[[FIVE]] : !py.dynamic into @x
# CHECK: %[[THREE:.*]] = constant(#py.int<3>)
# CHECK: store %[[THREE]] : !py.dynamic into @x

def foo4():
    global x
    try:
        while True:
            try:
                break
            finally:
                x = 5
    finally:
        x = 3


# CHECK-LABEL: @"foo4$impl[0]"
# CHECK: ^[[CONDITION:[[:alnum:]]+]]:
# CHECK: ^[[BODY:[[:alnum:]]+]]:
# CHECK: %[[FIVE:.*]] = constant(#py.int<5>)
# CHECK: store %[[FIVE]] : !py.dynamic into @x
# CHECK: br ^[[EXIT_BLOCK:[[:alnum:]]+]]

# CHECK: ^[[EXIT_BLOCK:[[:alnum:]]+]]:
# CHECK: %[[THREE:.*]] = constant(#py.int<3>)
# CHECK: store %[[THREE]] : !py.dynamic into @x

def foo5():
    global x
    try:
        try:
            pass
        finally:
            return
    finally:
        x = 3

# CHECK-LABEL: @"foo5$impl[0]"
# CHECK-NOT: ^{{[[:alnum:]]+}}
# CHECK: %[[THREE:.*]] = constant(#py.int<3>)
# CHECK: store %[[THREE]] : !py.dynamic into @x
