# RUN: pylir %s -o %t
# RUN: %t | FileCheck %s --match-full-lines

def random():
    return True


def foo():
    y = 0
    try:
        while random():
            if random():
                raise NameError
            y
    except NameError:
        y = 5

    return y


print(foo())
# CHECK: 5
